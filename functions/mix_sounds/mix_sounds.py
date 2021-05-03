# creator iman shahriari
# github: https://github.com/iman3sh
import numpy as np
import random
import subprocess
import os
import soundfile as sf
import copy

def mkdir(addr):
    try:
        os.stat(addr)
    except:
        os.mkdir(addr)

class create_dset():
    def __init__(self):
        self.dataset_root = '/home/iman/storage/code/dataset/speech/Complete_dataset/Persian/SampleDeepMine/wav'
        self.text_file_path = 'trans_FAR_ASR.lst'
        self.train_amount = 0.7
        self.test_amount = 0.15
        self.valid_amount = 0.15

        self.temp_addr = './temp'
        self.outputs_addr = './outputs'
        mkdir(self.temp_addr)
        mkdir(self.outputs_addr)

        needed_dir = ['train', 'test', 'valid']
        for ele in needed_dir:
            mkdir(os.path.join(self.outputs_addr, ele))
            mkdir(os.path.join(self.outputs_addr, ele, 's1'))
            mkdir(os.path.join(self.outputs_addr, ele, 's2'))
            mkdir(os.path.join(self.outputs_addr, ele, 'mix'))

    def split_train_test_valid(self):
        all_speakers = []
        self.train_speakers = []
        self.test_speakers = []
        self.valid_speakers = []
        for aline in open(self.text_file_path, "r"):
            all_speakers.append(aline.split()[0].split('/')[0])
        all_speakers = list(set(all_speakers))
        for speaker in all_speakers:
            if len(self.train_speakers) < 2 and self.train_amount > 0:
                self.train_speakers.append(speaker)
            elif len(self.test_speakers) < 2 and self.test_amount > 0:
                self.test_speakers.append(speaker)
            elif len(self.valid_speakers) < 2 and self.valid_amount > 0:
                self.valid_speakers.append(speaker)
            elif len(self.train_speakers)/len(all_speakers) < self.train_amount:
                self.train_speakers.append(speaker)
            elif len(self.test_speakers)/len(all_speakers) < self.test_amount:
                self.test_speakers.append(speaker)
            else:
                self.valid_speakers.append(speaker)

    def make_list_of_wav(self,list_of_speaker):
        list_of_wav = []
        for aline in open(self.text_file_path, "r"):
            if aline.split()[0].split('/')[0] in list_of_speaker:
                list_of_wav.append(aline.split()[0])
        return list_of_wav

    def _separate_list(self):
        self.train_wav_list = self.make_list_of_wav(self.train_speakers)
        self.test_wav_list = self.make_list_of_wav(self.test_speakers)
        self.valid_wav_list = self.make_list_of_wav(self.valid_speakers)


    def select_second_wav(self, first_wav):
        if first_wav in self.train_wav_list:
            candidate_waves = self.train_wav_list.copy()
        elif first_wav in self.test_speakers:
            candidate_waves = self.test_wav_list.copy()
        else:
            candidate_waves = self.valid_wav_list.copy()

        # remove all waves from that speaker which is exist in candidate_waves
        copy_candidate_waves = candidate_waves.copy()
        for wav in copy_candidate_waves:
            if wav.split('/')[0] == first_wav.split('/')[0]:
                candidate_waves.remove(wav)

        index = random.randint(0, len(candidate_waves)-1)
        return candidate_waves[index]

    def extract_proper_input(self, input1, input2):
        command = ['ffmpeg', '-y', '-i', input1, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                   os.path.join(self.temp_addr,'temp1.wav'), '-loglevel', 'quiet']
        subprocess.call(command)

        command = ['ffmpeg', '-y', '-i', input2, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                   os.path.join(self.temp_addr,'temp2.wav'), '-loglevel', 'quiet']
        subprocess.call(command)

    def cal_adjusted_rms(self, wav1_rms, snr):
        a = float(snr) / 20
        wav2_rms = wav1_rms / (10 ** a)
        return wav2_rms

    def add_noise(self, input1, input2, SNR):
        if len(input2) > len(input1):
            wav1 = input1
            wav2 = input2
            wav1_is_input1 = True
        else:
            wav1 = input2
            wav2 = input1
            wav1_is_input1 = False
        
        if wav1.dtype != 'float64':
            wav1 = wav1.astype(np.float64)
        if wav2.dtype != 'float64':
            wav2 = wav2.astype(np.float64)
        wav1_rms = np.sqrt(np.mean(np.square(wav1), axis=-1))

        start = random.randint(0, len(wav2) - len(wav1))
        divided_wav2_amp = wav2[start: start + len(wav1)]
        wav2_rms = np.sqrt(np.mean(np.square(divided_wav2_amp), axis=-1))

        adjusted_wav2_rms = self.cal_adjusted_rms(wav1_rms, SNR)

        adjusted_wav2_amp = divided_wav2_amp * (adjusted_wav2_rms / wav2_rms)
        mix_wav = (wav1 + adjusted_wav2_amp)
        # Avoid clipping noise
        max_float64 = np.finfo(np.float64).max
        min_float64 = np.finfo(np.float64).min
        if mix_wav.max(axis=0) > max_float64 or mix_wav.min(axis=0) < min_float64:
            if mix_wav.max(axis=0) >= abs(mix_wav.min(axis=0)):
                reduction_rate = max_float64 / mix_wav.max(axis=0)
            else:
                reduction_rate = min_float64 / mix_wav.min(axis=0)
            mix_wav = mix_wav * (reduction_rate)
            wav1 = wav1 * (reduction_rate)
            adjusted_wav2_amp = adjusted_wav2_amp * (reduction_rate)
        wav1 = wav1.astype(np.float64)
        adjusted_wav2_amp = adjusted_wav2_amp.astype(np.float64)
        mix_wav = mix_wav.astype(np.float64)
        if wav1_is_input1:
            return wav1, adjusted_wav2_amp, mix_wav
        if not wav1_is_input1:
            return adjusted_wav2_amp, wav1, mix_wav

    def run(self):
        self.split_train_test_valid()
        self._separate_list()
        i=0
        for aline in open(self.text_file_path, "r"):
            wav1 = aline.split()[0]
            if wav1 in self.train_wav_list:
                destination = 'train'
            elif wav1 in self.test_wav_list:
                destination = 'test'
            else:
                destination = 'valid'
            wav2 = self.select_second_wav(wav1)
            save_name = '_'.join(['-'.join(wav1.split('/')), '-'.join(wav2.split('/'))])+'.wav'
            self.extract_proper_input(os.path.join(self.dataset_root, wav1+'.wav'), os.path.join(self.dataset_root, wav2+'.wav'))

            input1, _ = sf.read(os.path.join(self.temp_addr,'temp1.wav'))
            input2, sr = sf.read(os.path.join(self.temp_addr,'temp2.wav'))
            snr = random.uniform(-5, 5)
            s1, s2, mix = self.add_noise(input1, input2, snr)
            sf.write(os.path.join(self.outputs_addr, destination, 's1', save_name),s1,sr)
            sf.write(os.path.join(self.outputs_addr, destination, 's2', save_name),s2,sr)
            sf.write(os.path.join(self.outputs_addr, destination, 'mix', save_name),mix,sr)
            i+=1
            # if i>10:
            #     break
            print(i)
        print('done!')

if __name__=='__main__':
    create_dataset = create_dset()
    create_dataset.run()









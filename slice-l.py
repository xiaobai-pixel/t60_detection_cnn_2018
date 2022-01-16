import numpy as np
from scipy.signal import lfilter
import gtg
import splweighting
import wave
import glob
import os
##configuration area
chunk_length = 4
chunk_overlap = 0.5
dir_str = r"/data2/queenie/IEEE2015Ace/dataset/Dev/Speech/Chromebook"
#file1 = "EM32_Building_Lobby_1_M6_s3_Fan_20dB.wav"
room_name = ['Building_Lobby','Lecture_Room_1','Lecture_Room_2','Meeting_Room_1','Meeting_Room_2','Office_1','Office_2']
save_dir = os.path.join("/data2/queenie/IEEE2015Ace/solution/DatasetProcessing",dir_str.split("/")[-1])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
##functions
def SPLCal(x):
    Leng = len(x)
    pa = np.sqrt(np.sum(np.power(x, 2))/Leng)
    p0 = 2e-5
    spl = 20 * np.log10(pa / p0)
    return spl

print(dir_str)
##main loop, process eahc file in dir
for file_name in glob.glob(dir_str+r"/*.wav"):
    f = wave.open(file_name, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(file_name)
    print(nchannels, sampwidth, framerate, nframes/framerate)
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(str_data, dtype=np.int16)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T
    audio_time = nframes/framerate
    chan_num = 0
    count = 0
    new_file_name = "%s" % (file_name.split("/")[-1].split(".")[0])
    ##process each channel of audio
    for audio_samples_np in wave_data:
        whole_audio_SPL = SPLCal(audio_samples_np)

        available_part_num = (audio_time-chunk_overlap)//(chunk_length - chunk_overlap)   #4*x - (x-1)*0.5 <= audio_time    x为available_part_num

        if available_part_num ==1:
            cut_parameters = [chunk_length]
        else:
            cut_parameters = np.arange(chunk_length, (chunk_length - chunk_overlap)*available_part_num+chunk_overlap, chunk_length)  # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）

        start_time = int(0)  # 开始时间设为0
        count = 0

        for t in cut_parameters:
            stop_time = int(t)  # pydub以毫秒为单位工作
            start = int(start_time*framerate)
            end = int((start_time+chunk_length)*framerate)
            audio_chunk = audio_samples_np[start:end]  # 音频切割按开始时间到结束时间切割

            ##ingore chunks with no audio
            chunk_spl = SPLCal(audio_chunk)
            if whole_audio_SPL - chunk_spl >=20:
                continue

            ##file naming
            npy_file_name = os.path.join(save_dir,new_file_name + '-' + str(count) + '_' + str(chan_num) + '.npy')
            count +=1
            print(npy_file_name)

            ##A weighting
            chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)

            ##gammatone
            chunk_gtg,cen_freq = gtg.gtg_processing(chunk_a_weighting, framerate)

            ##whitening
            chunk_result = chunk_gtg
            for i in range(21):
                chunk_gtg_tmp = chunk_gtg[i] - np.mean(chunk_gtg[i])
                chunk_result[i] = chunk_gtg_tmp/np.max(np.abs(chunk_gtg_tmp))
            ##plot
            #gtg.gtgplot(chunk_result, cen_freq, len(audio_chunk), framerate)#just for debug

            ##save file
            #print(chunk_result)
            result_h,result_w = chunk_result.shape
            #print(chunk_result)
            print("chunk_result shape:",(result_h,result_w))
            assert result_h==21
            assert result_w==1999
            np.save(npy_file_name, chunk_result)
            #audio_chunk.export(new_file_name, format="wav")  # 保存音频文件，t/2只是为了计数，根据步长改变。步长为5就写t/5
            start_time = start_time + chunk_length - chunk_overlap  # 开始时间变为结束时间前1s---------也就是叠加上一段音频末尾的4s

        chan_num = chan_num + 1
    print('----------------finish----------------')






#######################
# ultrafeedback_binarized 데이터를 squarelike/Gugugo-koen-7B-V1.1-AWQ 로 번역하기 위한 convert 프로그램
# FILE : gugugo-ultrafeedback_binarized.2023.1201.py
# date : 2023.1203
# 사용방법 : 
#
# comment : 
# - gugugo v1.1는 긴 문장 번역 시에 문장 생략 현상이 심함. 
#   대략 len(1000) 이면 조금씩 잘려나가가기 시작하고, len(2000) 넘어가면 사용 불가능.
#   그래서 문단을 한 줄 한 줄 잘라서 번역한 후에 다시 붙이기로 했음.
#
# - frequency_penalty=1.2 를 넣으면 마지막 단어 반복하는 현상은 많이 줄어듦
#######################

import torch
import textwrap
import datetime
import json
import os
import argparse   	# 내장모듈
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MAX_TOKEN = 4000

#######################
# 
#######################
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

#######################
# 
#######################
def save_json( ret_json ):
    with open(arg_output_filename, 'w', encoding='utf-8') as f:
        json.dump(ret_json, f, ensure_ascii=False, indent=4)
    print("=== 데이터 저장 완료. (%i)개" % len(ret_json) )

#######################
# 
#######################
def make_prompt(input_all):
    prompts = []
    for line in input_all:
        # line = line.replace('\n', '^^^n')
        prompts.append(f"### 영어: {line}</끝>\n### 한국어:")
    return prompts


#######################
# 
#######################
def translate_core( input ):
    texts = []
    texts.append( input )

    prompts = make_prompt(texts)
    sampling_params = SamplingParams(temperature=0.01, stop=["</끝>"], max_tokens=MAX_TOKEN
                                        , frequency_penalty=1.2 )
    ret = llm.generate(prompts, sampling_params)
    return ret[0].outputs[0].text.strip()

#######################
# gugugo는 문장이 길어지면, 라인을 없애거나 잘못 요약하는 일이 잦아서, 한 줄씩 번역하도록 수정
#######################
def translate( input ):
    temp = input.split( '\n' )
    ret = ""
    for line in temp:
        # print( "===line", line )
        ret_line = translate_core( line )
        ret = ret + ret_line + "\n"
    ret.strip( '\n')
    return ret

#########################
# argument를 파싱해서 가져옴
#########################
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', metavar='N', nargs='+',
                    help='filename')
    parser.add_argument('--restart_count', metavar='N', type=int, nargs='+',
                    help='restart_count')
    parser.add_argument('--output', metavar='N', nargs='+',
                    help='output')

    filename = parser.parse_args().filename
    output = parser.parse_args().output
    restart_count = parser.parse_args().restart_count

    return filename[0], output[0], restart_count[0]

#######################
# MAIN START
#######################
arg_filename, arg_output_filename, arg_restart_count = get_arguments()
directory, filename = os.path.split(arg_filename)

print( "\n\n" )
print( "directory:", directory )
print( "filename:", filename )
print( "output:", arg_output_filename)
print( "restart_count:", arg_restart_count)

# model_id = "/aicc/model/squarelike_Gugugo-koen-7B-V1.1-GPTQ"
model_id = "/han/model/squarelike_Gugugo-koen-7B-V1.1-AWQ"

from vllm import LLM, SamplingParams

llm = LLM(model="squarelike/Gugugo-koen-7B-V1.1-AWQ", quantization="awq", dtype="half")

ret_all = []
input_all = []
count = 0
break_flag = False

with open( arg_filename, 'r', encoding='utf-8') as file:
    input_all = json.load(file)

print( "===데이터갯수", len(input_all) )
ret_all_count = len(input_all)

input_small = input_all[arg_restart_count:-1]
count = arg_restart_count
err_count = 0

for line in input_small:
    print( "===", datetime.datetime.now().strftime( '%Y%m%d.%H%M'), "FILE :", filename, "#", count, "/", len(input_all), "err:", err_count, ", line['prompt']", line["prompt_id"] )
    line["prompt_kor"] = translate( line["prompt"] )
    print( "line[\"prompt_kor\"] ori len (%i), _kor len(%i)" % (len(line["prompt"]), len(line["prompt_kor"] )) )
    
    for i in range( 0, len( line["chosen"] ) ):
        if i == 0: 
            # line["prompt_kor"]는 총 4번 쓰인다.
            line["chosen"][i]["content_kr"] = line["prompt_kor"]
        else:
            buf = translate( line["chosen"][i]["content"] )
            line["chosen"][i]["content_kr"] = buf
    
        print( "content #%i ori len (%i)" % (i, len(line["chosen"][i]["content"])) )

    for i in range( 0, len( line["rejected"] ) ):
        if i == 0:
            line["rejected"][i]["content_kr"] = line["prompt_kor"]
        else:
            buf = translate( line["rejected"][i]["content"] )
            line["rejected"][i]["content_kr"] = buf

        # print( "rejected #%i ori len (%i), _kor len(%i)" % (i, len(line["rejected"][i]["content"]), len(buf)) )
        print( "content #%i ori len (%i)" % (i, len(line["rejected"][i]["content"])) )
    
    for i in range( 0, len( line["messages"] ) ):
        if i == 0:
            line["messages"][i]["content_kr"] = line["prompt_kor"]
        else:
            buf = translate( line["messages"][i]["content"] )
            line["messages"][i]["content_kr"] = buf

        # print( "messages #%i ori len (%i), _kor len(%i)" % (i, len(line["messages"][i]["content"]), len(buf)) )
        print( "content #%i ori len (%i)" % (i, len(line["messages"][i]["content"])) )
          
    ret_all.append( line )
    count = count + 1
    if count % 50 == 0:
        save_json( ret_all )
        
save_json( input_small )
print( "=== count", count, "/", ret_all_count)

#######################
# end of file : gugugo-ultrafeedback_binarized.2023.1201.py
#######################

from lib.pytracking.vot.vitcrt_stb import run_vot_exp
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_vot_exp('vit_crt', 'baseline', vis=False)

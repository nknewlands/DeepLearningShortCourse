#!/usr/bin/python
# ==============================================================================
# Divide and image directory into sample directory
# Etienne Lord - 2019
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Note: we expect a directory structure as:
# 
# indir/class_1
# indir/class_2
# as input..

import argparse
import numpy as np
import os
import shutil
from pathlib import Path


# DATA, % Training, % validation
def divides(data, ptraining, pvalidation):
  l=len(data)
  tq=((l)//100)*pvalidation
  tend=((l)//100) *ptraining
  vstart=tend
  vend=vstart+tq
  teststart=vend
  ret=[data[:tend],data[vstart:vend],data[teststart:]]
  print("Training  : " +str(len(ret[0]))+" ("+str(ptraining)+"%)")
  print("Validation: "+str(len(ret[1]))+" ("+str(pvalidation)+"%)")
  print("Test      : "+str(len(ret[2])))
  return (ret)
  
def create_set(datadir, ptrain, pval, seed, number):
  PATH = Path(datadir)
  train_path = PATH
  classes = [str(f).split(os.path.sep)[-1] for f in list(PATH.iterdir())]
  files = []
  print("Found those classes:")
  print(classes)
  for i in classes:
    paths =train_path/i
    print(i)
    j=0
    for f in list(paths.iterdir()):
      files.append(f)
      j+=1
    print(j)
  print(len(files))
  np.random.seed(seed)
  np.random.shuffle(files)
  
  if number>-1: 
    files=files[:number]
  
  (training,validation,test)=divides(files,ptrain,pval)
  
  shutil.rmtree(str(PATH)+"_training", ignore_errors=True)
  shutil.rmtree(str(PATH)+"_validation", ignore_errors=True)
  shutil.rmtree(str(PATH)+"_test", ignore_errors=True)
  
  os.makedirs(str(PATH)+"_training", exist_ok=True)
  os.makedirs(str(PATH)+"_validation", exist_ok=True)
  os.makedirs(str(PATH)+"_test", exist_ok=True)
  for f in test:
    p = Path(f)
    filename=p.parts[-1]
    pth=p.parts[-2]
    os.makedirs(str(PATH)+"_test"+"/"+pth, exist_ok=True)
    shutil.copy2(f,str(PATH)+"_test"+"/"+pth+"/"+filename)
  for f in validation:
    p = Path(f)
    filename=p.parts[-1]
    pth=p.parts[-2]
    os.makedirs(str(PATH)+"_validation"+"/"+pth, exist_ok=True)
    shutil.copy2(f,str(PATH)+"_validation"+"/"+pth+"/"+filename)
  for f in training:
    p = Path(f)
    filename=p.parts[-1]
    pth=p.parts[-2]
    os.makedirs(str(PATH)+"_training"+"/"+pth, exist_ok=True)
    shutil.copy2(f,str(PATH)+"_training"+"/"+pth+"/"+filename)    

if __name__ == "__main__":
# Parser variables
    parser = argparse.ArgumentParser(description='Create a training directory')
    parser.add_argument('-indir', type=str, help='Input dir for data', default='', required=True)
    parser.add_argument('-ptrain', type=int, help='percent training', default=70)
    parser.add_argument('-pval', type=int, help='percent validation', default=15)
    parser.add_argument('-n', type=int, help='Number of images (default: all)', default=-1)
    parser.add_argument('-seed', type=int, help='seed', default=42)
    args = parser.parse_args()
    create_set(args.indir, args.ptrain, args.pval, args.seed, args.n)
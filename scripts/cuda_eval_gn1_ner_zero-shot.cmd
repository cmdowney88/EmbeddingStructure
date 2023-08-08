executable = scripts/eval_ner.sh
arguments = "/home2/ngoldf txlm 0 configs/uralic_ner_zero-shot_xlmr-ots.yml"
getenv = true
output = output/running/uralic_ner_zero-shot_xlmr-ots.out
error = output/running/uralic_ner_zero-shot_xlmr-ots.err
log = output/running/uralic_ner_zero-shot_xlmr-ots.log
Requirements = (( machine == "patas-gn1.ling.washington.edu" ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
queue
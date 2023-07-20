executable = scripts/eval_ner.sh
arguments = "/home2/ngoldf txlm 0 configs/uralic_ner_equal-train_xlmr-ots.yml"
getenv = true
output = output/running/uralic_ner.out
error = output/running/uralic_ner.err
log = output/running/uralic_ner.log
Requirements = (( machine == "patas-gn1.ling.washington.edu" ))
request_GPUs = 1
transfer_executable = false
notification = always
+Research = true
request_memory = 5*1024
queue
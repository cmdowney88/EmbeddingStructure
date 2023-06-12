#!/bin/sh

OUTPUT_NAME=$1
INPUT_FILES=${@:2}
COMBINED_FILE="${OUTPUT_NAME}_combined.txt"
COMBINED_CLEANED_FILE="${OUTPUT_NAME}_combined_cleaned.txt"

# Make the combined data file blank, and remove the output file if it already exists
touch $COMBINED_FILE
echo -n > $COMBINED_FILE
rm -f $COMBINED_CLEANED_FILE

# Combine the data from the input files
for filename in $INPUT_FILES
do
    cat $filename >> $COMBINED_FILE
done

# Deduplicate and filter
# This package primarily wants to work with JSON configs, which doesn't play nicely with shell
# interpolation. Single quotes indicate the beginning of a literal JSON object. Variable
# interpolation is not allowed within single quotes, thus the ugly '["' components
opusfilter-cmd remove_duplicates --inputs '["'${COMBINED_FILE}'"]' --outputs '["'${OUTPUT_NAME}_dedup_tmp.txt'"]' --lowercase True --letters_only True
opusfilter-cmd filter --inputs '["'${OUTPUT_NAME}_dedup_tmp.txt'"]' --outputs '["'${COMBINED_CLEANED_FILE}'"]' --filters \
'[{"LengthFilter": {"min_length": 2}}, {"AverageWordLengthFilter": {"max_length": 16}}, {"LongWordFilter": {"threshold": 32}}, {"AlphabetRatioFilter": {"threshold": 0.5}}]'

# Remove temporary file
rm -f *_tmp.txt

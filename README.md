# prune
A neural network appraoch to DNA contamination removal

Uses the Keras Libary for neural network creation to filter specific sequences from a fastq/fasta

# Usage

Provide a txt file containing the seqeunces to be serached for and a fastq as your background

### Preparation

1. Clone or copy the files, one easy way to do this is git clone https://github.com/JonasDauster/prune

2. Provide a set of normal, clean sequences that do not contain the sequences you want to search for or simply your  (as fastq, for reference see example data)

3. Provide the sequences you want to search for (as txt, for reference see example data)

4. Install all requiered packages, i recommend conda for that ( all python 3 versions should work fine )

### Training
1. Change the files in TrainingDataCreation.py accordingly and run it

2. Do the same with CNN.py, but before you let it run fix the python hash seed. One easy way(with linux) is to use this setup with an command line interface:  __PYTHONHASHSEED=6 python3 CNN.py__ . You can use any number behind PYTHONHASHSEED= , just choose the same when running the next steps.

### Testing/Running

1. For testing either generate a new training file and run it with DataLoad.py (use the same hashseed as for CNN.py) or use the validation_split method built in keras. 

2.For running on data to classify

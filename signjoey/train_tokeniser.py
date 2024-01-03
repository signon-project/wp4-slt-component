import os
import gzip
import pickle
import sentencepiece as spm

def load_dataset_file(filename): 
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
    return loaded_object

data_path = '/ikerlariak/anunez038/slt/data/PHOENIX2014T/phoenix14t.pami0.'
tokeniser_path = '../data/tokeniser_cngt/'
tokeniser_prefix = 'tok'
vocabulary_file = 'vocabulary.txt'
vocabulary_size = 3000
input_sentence_size = 8000
language_codes = ['<DE>']

corpus_file = '/ikerlariak/anunez038/sl_data/cngt-corpus-text.txt'

if not os.path.exists(tokeniser_path):
    os.makedirs(tokeniser_path)

# Hemen tokenizatzailea entrenatzeko behar dituzun esaldiak kargatzen dituzu "lines" array batean
""" lines = []
for data_set in ['dev', 'train']:
    data_dict = load_dataset_file(data_path + data_set) # hau da neccam-en kodean erabiltzen den funtzioa datuak kargatzeko, moldatu gure datuak kargatzeko
    lines += [data_dict[i]['text'].strip() for i in range(len(data_dict))] """

# "Lines" hori fitxategi batera pasatu (zuzenean pausu batean egin daiteke dena, egia esan)
""" corpus_file = 'temp.txt'
with open(corpus_file, 'w') as f:
    for line in lines:
        f.write(line.lower() + '\n') """

# Honekin entrenatzen duzu tokenizatzaile berria
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix=tokeniser_path + tokeniser_prefix, # prefix hau nahi duzuna, gero modeloa eta hiztegia honekin hasiko dira
    vocab_size=vocabulary_size,
    shuffle_input_sentence=True,
    input_sentence_size=input_sentence_size,
    num_sub_iterations=10,
    treat_whitespace_as_suffix=True,
    unk_piece='<unk>', 
    bos_piece='<s>', 
    eos_piece='</s>', 
    pad_piece='<pad>', 
    pad_id=3, # being 0, 1 and 2 the unk, bos and eos tokens, respectively
    #control_symbols=','.join(language_codes) # hau gehitu behar da hizkuntza berri bat sartuz gero
)   

# Generate new vocabulary
os.system(
    'spm_encode --model={} --generate_vocabulary < {} > {}'.format(
        tokeniser_path + tokeniser_prefix + '.model',
        corpus_file,
        tokeniser_path + vocabulary_file
    ))

# Remove the \t character and replace it by a blank space
os.system("sed -ie 's/\t/ /g' {}".format(
    tokeniser_path + vocabulary_file)
)   

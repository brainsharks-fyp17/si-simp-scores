import sentencepiece as spm

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
spm.SentencePieceTrainer.train('--input=tokenized_shard_100000.txt --normalization_rule_tsv=rules.tsv '
                               '--model_prefix=m --vocab_size=3000 --character_coverage=1.0 --model_type=bpe')

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# encode: text => id
print(sp.encode_as_pieces('බිරිඳගෙන් තමාට තර්ජන ඇති බවට සැමියා විසින් පොලිසියට පැමිණිලි කිරීමෙන්'))
print(sp.encode_as_ids('බිරිඳගෙන් තමාට තර්ජන ඇති බවට සැමියා විසින් පොලිසියට පැමිණිලි කිරීමෙන්'))



# Assumes that m.model is stored in non-Posix file system.
serialized_model_proto = tf.gfile.GFile('m.model', 'rb').read()

sp = spm.SentencePieceProcessor()
sp.load_from_serialized_proto(serialized_model_proto)

print(sp.encode_as_pieces('this is a test'))
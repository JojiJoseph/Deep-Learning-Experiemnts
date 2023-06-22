from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE, Unigram
from tokenizers.trainers import WordPieceTrainer
from tokenizers.trainers import BpeTrainer, UnigramTrainer

tokenizer = Tokenizer(WordPiece())
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["1661-0.txt"], trainer=trainer)
tokens = tokenizer.encode("I like playing cricket")

print(tokens.tokens)

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["1661-0.txt"], trainer=trainer)
tokens = tokenizer.encode("I like playing cricket")

print(tokens.tokens)

tokenizer = Tokenizer(Unigram())
trainer = UnigramTrainer()#special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["1661-0.txt"], trainer=trainer)
tokens = tokenizer.encode("I like        playing cricket")

print(tokens.tokens)
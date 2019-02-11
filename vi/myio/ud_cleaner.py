'''
remove X_X line in ud treebank
'''
import os.path

lang_dir = 'UD_Spanish'
lang = 'es'

data_dir = '/Users/boon/code/study/corpora/Universal Dependencies 1.4/ud-treebanks-v1.4'
target_dir = '../../data/ud'

train_fr = os.path.join(data_dir, lang_dir, lang+'-ud-train.conllu')
dev_fr = os.path.join(data_dir, lang_dir, lang+'-ud-dev.conllu')
test_fr = os.path.join(data_dir, lang_dir, lang + '-ud-test.conllu')

train_fw = os.path.join(target_dir, lang+'-ud-train_clean.conllu')
dev_fw = os.path.join(target_dir, lang+'-ud-dev_clean.conllu')
test_fw = os.path.join(target_dir, lang + '-ud-test_clean.conllu')

for fr, fw in zip((train_fr, dev_fr, test_fr), (train_fw, dev_fw, test_fw)):
    fpr = open(fr)
    fpw = open(fw, 'w')
    for line in fpr:
        if len(line.split()) > 0 and '-' in line.split()[0]:
            continue
        fpw.write(line)
    fpr.close()
    fpw.close()
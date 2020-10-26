# -*- coding: utf-8 -*-

# This code is from https://github.com/abisee/cnn-dailymail.git

import sys
import os
import hashlib
import struct
import subprocess
import collections
from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "dataset/url_lists/all_train.txt"
all_val_urls = "dataset/url_lists/all_val.txt"
all_test_urls = "dataset/url_lists/all_test.txt"

cnn_tokenized_stories_dir = "dataset/cnn_stories_tokenized"
dm_tokenized_stories_dir = "dataset/dm_stories_tokenized"
finished_files_dir = "dataset/finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = 'dataset/finished_files/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        # for set_name in ['train']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def get_source_target_txt(zipped_source_target):
    target = zipped_source_target[1].strip()
    target = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in [target]])

    return zipped_source_target[0].strip(), target


def write_to_bin(zipped_soruce_target_file, out_file, makevocab=False, sents_count=10000, start_indx=0):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    # print("Making bin file for URLs listed in %s..." % url_file)
    # url_list = read_text_file(url_file)
    # url_hashes = get_url_hashes(url_list)
    # story_fnames = [s + ".story" for s in url_hashes]
    # num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for i, item in enumerate(zipped_soruce_target_file):
            if i < start_indx:
                continue
            if i > sents_count + start_indx:
                break
            print(out_file, i)
            source_txt, target_txt = get_source_target_txt(item)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['source_txt'].bytes_list.value.extend([bytes(source_txt, encoding='utf-8')])
            tf_example.features.feature['target_txt'].bytes_list.value.extend([bytes(target_txt, encoding='utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = source_txt.split(' ')
                abs_tokens = target_txt.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


def create_dataset(source_addr, target_addr):
    source_txts = open(source_addr)
    target_txts = open(target_addr)
    source_target = zip(source_txts, target_txts)
    write_to_bin(source_target, out_file=os.path.join(finished_files_dir, 'train.bin'), makevocab=True,
                 sents_count=4530000, start_indx=0)
    write_to_bin(source_target, out_file=os.path.join(finished_files_dir, 'test.bin'), sents_count=1000,
                 start_indx=4530000)
    write_to_bin(source_target, out_file=os.path.join(finished_files_dir, 'val.bin'), sents_count=10000,
                 start_indx=4531000)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()


if __name__ == '__main__':
    create_dataset(source_addr='/content/data/input.txt', target_addr='/content/data/output.txt')

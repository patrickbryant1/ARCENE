'''
This is a deep CNN that trains and evaluates on the ARCENE dataset which main objective is to
distinguish cancer versus normal patterns from massspectrometric data. The Python programming language
in combination with TensorFlow, which uses a highly efficient C++ backend for computations.
'''

#Read files
#Queue of files to read
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
reader = tf.TextLineReader() #reads lines
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(    
  value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])


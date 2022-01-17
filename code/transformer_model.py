import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 200
		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

		# Define english and french embedding layers:
		self.fr_embedding = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size)
		self.en_embedding = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size)

		# Create positional encoder layers
		self.fr_position = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
		self.en_position = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.fr_encoder = transformer.Transformer_Block(self.embedding_size, False)
		self.en_decoder = transformer.Transformer_Block(self.embedding_size, True)

		# Define dense layer(s)
		self.dense1 = tf.keras.layers.Dense(200, activation = 'relu')
		self.dense2 = tf.keras.layers.Dense(self.english_vocab_size, activation = 'softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		#1) Add the positional embeddings to french sentence embeddings
		fr_input = self.fr_position(self.fr_embedding(encoder_input))

		#2) Pass the french sentence embeddings to the encoder
		out = self.fr_encoder(fr_input)

		#3) Add positional embeddings to the english sentence embeddings
		en_input = self.en_position(self.en_embedding(decoder_input))

		#4) Pass the english embeddings and output of your encoder, to the decoder
		out = self.en_decoder(en_input, out)

		#5) Apply dense layer(s) to the decoder out to generate probabilities
		out = self.dense1(out)

		return self.dense2(out)

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		return tf.reduce_sum(loss * mask)

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)

#include "pch.h"
#include "RNN_cell.h"
#include <assert.h>
#include <cmath>

// 权重初始化
void randInit(std::vector<std::vector<FLOAT>> &weights, unsigned dim1, unsigned dim2) {
	weights.resize(dim1);

	for (int i = 0; i < weights.size(); ++i) {
		weights[i].resize(dim2);
		for (int j = 0; j < weights[i].size(); ++j) {
			weights[i][j] = randVal(1);
		}
	}
}

std::vector<FLOAT> dot(const std::vector<std::vector<FLOAT>> &w, const std::vector<FLOAT> &v) {
	assert(v.size() == w.size(), "wrong dims!");

	std::vector<FLOAT> ret;
	for (int i = 0; i < w[0].size(); ++i) {
		FLOAT sum = 0;
		for (int j = 0; j < w.size(); ++j) {
			sum += w[j][i] * v[j];
		}
		ret.push_back(sum);
	}

	return ret;
}

std::vector<std::vector<FLOAT>> dot(const std::vector<std::vector<FLOAT>> &w, FLOAT v) {
	std::vector<std::vector<FLOAT>> ret(w);
	for (int i = 0; i < w[0].size(); ++i) {
		for (int j = 0; j < w.size(); ++j) {
			ret[j][i] = w[j][i] * v;
		}
	}

	return ret;
}

FLOAT sigmoid(FLOAT x) {
	return 1 / (1 + expf(-x));
}

FLOAT sigmoid_derivative(FLOAT y) {
	return y * (1 - y);
}

FLOAT tanH(FLOAT x) {
	return 0;
}

FLOAT activate(FLOAT z) {
	return sigmoid(z);
}

RNN_cell::RNN_cell(unsigned inputSize, unsigned hiddenSize, unsigned outputSize)
{
	randInit(weightsOfInput, inputSize, hiddenSize);
	randInit(weightsOfHidden, hiddenSize, hiddenSize);
	randInit(weightsOfOutput, hiddenSize, outputSize);

	historyHiddenOutput.push_back(std::vector<FLOAT>(hiddenSize));

	guess.resize(BINARY_NUM);
	guess_len = BINARY_NUM - 1;
}


RNN_cell::~RNN_cell()
{
}

FLOAT RNN_cell::forward(const std::vector<FLOAT> &input)
{
	if (input.empty())
		return 0;

	std::vector<FLOAT> output1 = dot(weightsOfInput, input);
	std::vector<FLOAT> output2 = dot(weightsOfHidden, historyHiddenOutput.back());
	
	assert(output1.size() == output2.size());

	std::vector<FLOAT> hidden_output;
	for (int i = 0; i < output1.size(); ++i) {
		hidden_output.push_back(activate(output1[i] + output2[i]));
	}

	std::vector<FLOAT> output = dot(weightsOfOutput, hidden_output);

	// 记录上一位的hidden output，用于下一位计算
	historyHiddenOutput.push_back(hidden_output);

	FLOAT o = activate(output[0]);
	guess[guess_len] = std::roundf(o);
	--guess_len;

	return o;
}

void RNN_cell::train(const std::vector<std::vector<FLOAT>> &inputs, std::vector<FLOAT> y) {
	std::vector<FLOAT> output_deltas;

	for (int i = inputs.size() - 1; i >= 0; --i) {
		FLOAT y_pred = forward(inputs[i]);
		FLOAT output_error = y[i] - y_pred;
		output_deltas.push_back( output_error * sigmoid_derivative(y_pred) );	
	}

	std::vector<FLOAT> future_hidden_layer_delta(historyHiddenOutput[0].size());

	for (int i = 0; i < inputs.size(); ++i) {
		std::vector<FLOAT> x = inputs[i];
		std::vector<FLOAT> hidden_output = historyHiddenOutput[historyHiddenOutput.size() - i - 1];
		std::vector<FLOAT> pre_hidden_output = historyHiddenOutput[historyHiddenOutput.size() - i - 2];

		FLOAT output_delta = output_deltas[output_deltas.size() - i - 1];
		std::vector<FLOAT> hidden_delta_1 = dot(weightsOfHidden, future_hidden_layer_delta);
		std::vector<std::vector<FLOAT>> hidden_delta_2 = dot(weightsOfOutput, output_delta);


	}
}


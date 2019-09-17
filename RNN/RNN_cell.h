#pragma once
#include <vector>
#include "Macro.h"

class RNN_cell
{
public:
	RNN_cell(unsigned inputSize, unsigned hiddenSize, unsigned outputSize);
	~RNN_cell();

	FLOAT forward(const std::vector<FLOAT> &inputs);
	void train(const std::vector<std::vector<FLOAT>> &inputs, std::vector<FLOAT> y);
	
	std::vector<FLOAT> guess;
	int guess_len;

	   
private:	
	std::vector<std::vector<FLOAT>> weightsOfInput;
	std::vector<std::vector<FLOAT>> weightsOfHidden;
	std::vector<std::vector<FLOAT>> weightsOfOutput;

	// records of hidden outputs
	// һ�ε�����һ��������м����ز���
	std::vector<std::vector<FLOAT>> historyHiddenOutput;

};


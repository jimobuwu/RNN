// RNN.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "RNN_cell.h"
#include "time.h"

std::vector<FLOAT> int2Bin(int num) {
	std::vector<FLOAT> ret(BINARY_NUM, 0);
	int index = BINARY_NUM - 1;
	while (num) {
		ret[index] = num & 0x1;
		num >>= 1;
		--index;
	}

	return ret;
}

int main()
{
    std::cout << "Hello World!\n"; 
	srand(time(NULL));

	RNN_cell * cell = new RNN_cell(2, 16, 1);

	for (int i = 0; i < 20000; ++i) {
		cell->guess_len = BINARY_NUM - 1;

		int a = rand() % 127;
		std::vector<FLOAT> a_bin = int2Bin(a);
		int b = rand() % 127;
		std::vector<FLOAT> b_bin = int2Bin(b);
		int c = a + b;
		std::vector<FLOAT> c_bin = int2Bin(c);

		std::vector< std::vector<FLOAT>> inputs;
		for (int i = 0; i < BINARY_NUM; ++i) {
			std::vector<FLOAT> oneInput;
			oneInput.push_back(a_bin[i]);
			oneInput.push_back(b_bin[i]);
			inputs.push_back(oneInput);
		}		

		cell->train(inputs, c_bin);
	}



}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

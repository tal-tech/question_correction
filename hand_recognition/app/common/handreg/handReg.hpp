/*
 * handReg.hpp
 *
 *  Created on: 2020年8月10日
 *      Author: ym
 */

#ifndef SRC_HANDREG_HPP_
#define SRC_HANDREG_HPP_

#include <iostream>
#include <vector>
#include <string>

class HandReg {
public:
	HandReg(bool is_formula);
	~HandReg();
public:
	int pretreatmentOfRotate(const std::string &inMat, std::string &outMat);
	int pretreatmentOfDetect(const std::string &angleOfRotate, std::string &detectMatStr);
	void detectStruct(const std::string &outputOfDetect);
	int beforeReg(std::vector<std::string> &allFormulaMats, std::vector<std::string> &handMats);
	void afterFormulaReg(const std::string tempMatStr);
	void afterTextReg(const std::string tempMat);
	void afterReg();
	void operateColumn();
	std::string combineJson();
private:
	void *user_data;
};


#endif /* SRC_HANDREG_HPP_ */

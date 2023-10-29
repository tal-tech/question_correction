%module handreg

%{
#include "handReg.hpp"
%}

%include stl.i
%include typemaps.i 
%include std_string.i
/* instantiate the required template specializations */
namespace std {
    %template(strVector)    vector<std::string>;
}

%apply std::string & INOUT {std::string &outMat};
%apply std::string & INOUT {std::string &detectMatStr};
%apply std::vector<std::string>& INOUT {std::vector<std::string> &allFormulaMats};
%apply std::vector<std::string>& INOUT {std::vector<std::string> &handMats};
/* Let's just grab the original header file here */
%include "handReg.hpp"

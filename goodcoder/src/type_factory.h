/// @file type.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_H
#define GOODCODER_SRC_TYPE_H

#include"parse_struct.h"
#include"type_spec.h"
#include"type.h"
namespace goodcoder{
class TypeFactory{
public:
    Type* createType(Form f, const std::string& str, int len){
        switch(f){
            case Form::INT:
                return new TypeInt(str, 1);
            case Form::FLOAT:
                return new TypeFloat(str, 1);
            case Form::FLOAT_ARRAY:
                return new TypeFloat(str, len);
            case Form::INT_ARRAY:
                return new TypeInt(str, len);
            case Form::STRING:
                return new TypeStr(str, len);
                //TODO user
            default:
                return nullptr;
        }
    }
private:
};
}
#endif
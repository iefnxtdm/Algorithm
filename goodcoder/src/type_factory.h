/// @file type.h
/// @brief 
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_TYPE_FACTORY_H
#define GOODCODER_SRC_TYPE_FACTORY_H

#include"parse_struct.h"
#include"type_spec.h"
namespace goodcoder{
class TypeFactory{
public:
    MyType* create_type(Form f, const std::string& str, int len){
        MyType* tp = nullptr;
        switch (f){
            case Form::INT:
                tp = dynamic_cast<MyType*>(new TypeInt(str, 1));
                break;
            case Form::FLOAT:
                tp = dynamic_cast<MyType*>(new TypeFloat(str, 1));
                break;
            case Form::FLOAT_ARRAY:
                tp = dynamic_cast<MyType*>(new TypeFloat(str, len));
                break;
            case Form::INT_ARRAY:
                tp = dynamic_cast<MyType*>(new TypeInt(str, len));
                break;
            case Form::STRING:
                return new TypeStr(str, len);
            case Form::USER:
                return new TypeUser(str);
            default:
                return nullptr;
        }
        return tp;
    }

};
}
#endif
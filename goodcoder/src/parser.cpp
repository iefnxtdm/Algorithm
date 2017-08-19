/// @file parser.cpp
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#include<vector>
#include<string>
#include"parser.h"
using std::vector;
using std::string;
namespace goodcoder{

    int Parser::parse(const std::string& line){
        vector<string> res;
        StringUtil::splits_tring(line, "\t", &res, 100);
        for(size_t i = 0; i < res.size(); ++i){
            const string& tmp = res[i];
            Type* store;
            if (judge_array(tmp, store)){
                _store->push_back(store);
                continue;   
            }else if(judge_float(tmp, store)){
                _store->push_back(store);
                continue;
            }else if(judge_int(tmp, store)){
                _store->push_back(store);
                continue;
            }else if(judge_str(tmp, store)){
                _store->push_back(store);
                continue;
            }
            //comlog warning TODO
        }
        return 0;
    }
    int Parser::judge_int(const std::string& str, Type* store){
        bool flag = true;
        for(size_t i = 0; i < str.size(); ++i){
            if(str[i] >= '0' && str[i] <= '9'){
                continue;
            }
            flag = false;
        }
        if (flag){
            store = new TypeInt(str, 1);
        }
        return flag? 0: -1;
    }
    int Parser::judge_str(const std::string& str, Type* store) {
        bool flag = true;
    }
    int Parser::judge_float(const std::string& str, Type* store){

    }
    int Parser::judge_array(const std::string& str, Type* store){

    }
    //TODO
    int Parser::judge_user(const std::string& str, Type* store){

    }

}

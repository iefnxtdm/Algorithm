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
            Form f; 
            if (!judge_array(tmp, f)){
                continue;   
            }else if(!judge_float(tmp)){
                f = Form::FLOAT;
                continue;
            }else if(!judge_int(tmp)){
                f = Form::INT;
                continue;
            }else if(!judge_str(tmp)){
                f = Form::STRING;
                continue;
            }
            //comlog warning TODO
        }
        return 0;
    }
    int Parser::judge_int(const std::string& str){
        bool flag = true;
        for(size_t i = 0; i < str.size(); ++i){
            if(str[i] >= '0' && str[i] <= '9'){
                continue;
            }
            flag = false;
            break;
        }
        return flag? 0: -1;
    }
    int Parser::judge_str(const std::string& str) {
        return 0;
    }
    int Parser::judge_float(const std::string& str){
        std::string::size_type pos1 = str.find_first_of(".");
        std::string::size_type pos2 = str.find_last_of(".");
        if(pos1 == std::string::npos || pos1 != pos2 || 
            pos1 == 0 || pos1 == str.size() - 1){
            return -1;
        }
        bool flag = true;
        for(size_t i = 0; i < str.size(); ++i){
            if ((str[i] >= '0' && str[i]<= '9') || str[i] == '.'){
                continue;
            }
            flag = false;
            break;
        }
        return flag? 0: -1;
    }
    int Parser::judge_array(const std::string& str, Form& f){
        auto pos1 = str.find_first_of(":");
        auto pos2 = str.find_last_of(":");
        if (pos1 == std::string::npos || pos1 != pos2){
            return -1;
        }
        auto len_str = str.substr(0, pos1);
        int len = 0;
        if (judge_int(len_str) == 0){
            len = StringUtil::str_to_int(len_str);
        }else{
            return -1;
        }
        vector<string> res;
        auto arr_str = str.substr(pos1 + 1);
        StringUtil::splits_tring(arr_str, ",", &res, 100);
        int flag = 0; // 1:int 2:float
        for (size_t i = 0; i < res.size(); i++){
            if (i == 0){
                if(judge_float(res[i])){
                    flag = 2;
                }else if(judge_int(res[i])){
                    flag = 1;
                }
                return -1;
            }
            if (flag == 2 && !judge_float(res[i])){
                continue;
            }
            if(flag == 1 && !judge_int(res[i])){
                continue;
            }
            return -1;
        }
        return 0;
    }
    //TODO
    int Parser::judge_user(const std::string& str){

    }

}

/// @file string_util.h
/// @brief
/// @author sunminqi@baidu.com
/// @version 1.0
/// @date 2017-07-31
#ifndef GOODCODER_SRC_STRING_UTIL_H
#define GOODCODER_SRC_STRING_UTIL_H
#include<vector>
#include<string>
#include<sstream>
class StringUtil{
public:

    /// @brief SplitString �ָ��ַ�
    ///
    /// @param src
    /// @param delimiter
    /// @param fields
    /// @param field_num
    ///
    /// @return
    static size_t splits_tring(const std::string& src,
                              const std::string& delimiter,
                              std::vector<std::string>* fields,
                              int field_num) {
        fields->clear();
        size_t previous_pos = 0;
        size_t current_pos = 0;
        while (static_cast<int>(fields->size()) < field_num &&
                (current_pos = src.find(delimiter, previous_pos)) != std::string::npos) {
            fields->push_back(src.substr(previous_pos, current_pos - previous_pos));
            previous_pos = current_pos + delimiter.length();
        }

        // Add the last string
        // If the last string is delimiter, add emyty string to fields too.
        if (previous_pos <= src.length()) {
            fields->push_back(src.substr(previous_pos));
        }

        return fields->size();
    }

    /// @brief str_to_int 
    ///
    /// @param str
    ///
    /// @return
    static int str_to_int(const std::string& str){
        std::istringstream ss(str);
        int num;
        ss>>num;
        return num;
    }

    /// @brief str_to_float ת��Ϊfloat
    ///
    /// @param str
    ///
    /// @return
    static float str_to_float(const std::string& str){
        std::istringstream ss(str);
        float num;
        ss>>num;
        return num;
    }

};
}
#endif

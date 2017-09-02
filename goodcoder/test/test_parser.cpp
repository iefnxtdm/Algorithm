#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <bmock.h>
#include <iostream>
#include <algorithm>
#include "src/parser.h"

using std::cout;
using ::testing::Return;
using ::testing::_;
using ::testing::Invoke;
using ::testing::SetArgumentPointee;
using ::testing::SetArgPointee;
using ::testing::SetArgReferee;
using ::testing::SetArrayArgument;
using ::testing::StrEq;
using ::testing::DoAll;

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace goodcoder {

/**
 * @brief
**/
class TestParserSuite : public ::testing::Test{
    protected:
        TestParserSuite(){};
        virtual ~TestParserSuite(){};
        virtual void SetUp() {
            //Called befor every TEST_F(test_MultiMerge_merge_mbr_suite, *)
        };
        virtual void TearDown() {
            //Called after every TEST_F(test_MultiMerge_merge_mbr_suite, *)
        };
};

/**
 * @brief
 * @begin_version
**/
TEST_F(TestParserSuite, case_float)
{
    Parser parser;
    parser.parse(std::string("3.5"));
    const std::vector<MyType*>& res = parser.get_data();
    void* val = nullptr;
    int len = 0;
    Form form = NONE;
    res[0]->get_val(val, len, form);
    EXPECT_TRUE(form == FLOAT);
    EXPECT_EQ(len, 1);
    float x = *(static_cast<float*>(val));
    EXPECT_EQ(x, 3.5);
}
TEST_F(TestParserSuite, case_int)
{
    Parser parser;
    parser.parse("-345687");
    const std::vector<MyType*>& res = parser.get_data();
    void* val = nullptr;
    int len = 0;
    Form form = NONE;
    res[0]->get_val(val, len, form);
    EXPECT_TRUE(form == INT);
    EXPECT_EQ(len, 1);
    int x = *(static_cast<int*>(val));
    EXPECT_EQ(x, -345687);
}
TEST_F(TestParserSuite, case_array)
{
    Parser parser;
    parser.parse("3:2.3,6.444,4.242123");
    const std::vector<MyType*>& res = parser.get_data();
    void* val = nullptr;
    int len = 0;
    Form form = NONE;
    res[0]->get_val(val, len, form);
    EXPECT_TRUE(form == FLOAT);
    EXPECT_EQ(len, 3);
    const float* p = static_cast<float*>(val);
    bool flag = fabs(p[0] - 2.3) < 0.00001?true:false;
    EXPECT_TRUE(flag);
    flag = fabs(p[1] - 6.444) < 0.00001?true:false;
    EXPECT_TRUE(flag);
    flag = fabs(p[2] - 4.242123) < 0.00001?true:false;
    EXPECT_TRUE(flag);
}
TEST_F(TestParserSuite, case_string)
{
    Parser parser;
    parser.parse(std::string("string"));
    const std::vector<MyType*>& res = parser.get_data();
    void* val = nullptr;
    int len = 0;
    Form form = NONE;
    res[0]->get_val(val, len, form);
    EXPECT_TRUE(form == STRING);
    EXPECT_EQ(len, 6);
    std::string a = *(static_cast<std::string*>(val));
    EXPECT_TRUE(a == "string");
}
TEST_F(TestParserSuite, case_user)
{
    Parser parser;
    parser.parse(std::string("232313-342523"));
    const std::vector<MyType*>& res = parser.get_data();
    void* val = nullptr;
    int len = 0;
    Form form = NONE;
    res[0]->get_val(val, len, form);
    EXPECT_TRUE(form == USER);
    EXPECT_EQ(len, 2);
    const int* p = static_cast<int*>(val);
    EXPECT_EQ(p[0], 232313);
    EXPECT_EQ(p[1], 342523);
}
}

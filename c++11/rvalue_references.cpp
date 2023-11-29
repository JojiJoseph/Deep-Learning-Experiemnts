#include <iostream>

using namespace std;

template <typename T>
void modify(T&& x){
    x = x + x;
}

int main() {
    int x = 1;
    modify(x);
    cout << x << endl;
    string str = "hello ";
    modify(str);
    cout << str << endl;
    modify(1);
    modify(string("hello"));
    return 0;
}
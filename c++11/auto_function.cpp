#include <iostream>

using namespace std;


auto add = [](auto a, auto b) -> auto { return a + b; };

int main() {
    cout << add(1, 2) << endl;
    cout << add(1.1, 2.2) << endl;
    cout << add(1.1, 2) << endl;
    cout << add(string("Hello "), "World") << endl;
    cout << add("Hello", 1) << endl;
}
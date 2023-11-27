#include <iostream>
using namespace std;

int main() {
    auto hello_world = []() { std::cout << "Hello World!" << std::endl; };
    hello_world();
    int x = 0;
    cout << "x = " << x << endl;
    auto inc1 = [&x]() { x = x+1; }; // Capture x by reference
    inc1();
    cout << "x = " << x << endl;
    auto inc2 = [&]() { x = x+1; }; // Capture everything by reference
    inc2();
    cout << "x = " << x << endl;
    auto print_x = [=] { cout << "x = " << x << endl; }; // Capture everything by value
    print_x();
    auto print_x2 = [x]() { cout << "x = " << x << endl; }; // Capture x by value
    print_x2();

    auto add = [](int x, int y) { return x+y; };
    cout << "add(1, 2) = " << add(1, 2) << endl;
    auto add2 = [](float x, float y) -> float { return x+y; };
    cout << "add2(1.1, 2.2) = " << add2(1.1, 2.2) << endl;
    return 0;
}
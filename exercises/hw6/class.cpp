#include <iostream>
#include <stdio.h>

#define N 2

class Managed
{
public:
    Managed()
    {
        std::cout << "Managed constructor" << std::endl;
    }
    void * operator new(size_t len)
    {
        std::cout << "new operator" << std::endl;
        void * ptr;
        ptr = ::operator new(len);
        return ptr;
    }

    void operator delete(void * ptr)
    {
        std::cout << "delete operator" << std::endl;
        ::operator delete(ptr);
    }
};

class umString : public Managed
{
public:
    char * data;
    int len;
};

class dataElem : public Managed
{
public:
    int key;
    umString name;
};

int main()
{
    std::cout << "Hello World" << std::endl;
    dataElem * data = new dataElem;
    //the new operator of dataElem is overloaded
    //And it will call the new operator of "key" and "name"
    //Managed * m = new Managed;
    std::cout << "data -> name.len is :" <<  data -> name.len << std::endl;
    return 0;
}
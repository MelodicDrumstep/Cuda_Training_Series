#include <cuda_runtime.h>
#include <iostream>

#define N 1000


class Managed
{
public:
    void * operator new(size_t len)
    {
        std::cout << "new operator" << std::endl;
        void * ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }
    void operator delete(void * ptr)
    {
        std::cout << "delete operator" << std::endl;
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

class umString : public Managed
{
public:
    umString(const umstring &s)
    {
        len = s.len;
        cudaMallocManaged(&data, len);
        memcpy(data, s.data, len);
    }

private:
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
    dataElem * data = new dataElem[N];
    //the new operator of dataElem is overloaded
    //And it will call the new operator of "key" and "name"
    
    return 0;
}
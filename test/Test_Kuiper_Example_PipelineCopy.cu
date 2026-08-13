#include "Kuiper_Example_PipelineCopy.h"

int main()
{
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    Kuiper_Example_PipelineCopy_hf(a);
    printf("%f %f %f %f\n", a[0], a[1], a[2], a[3]);
    return 0;
}

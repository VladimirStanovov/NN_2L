#include <time.h>
#include <stdlib.h>

int IntRandom(int target)
{
    return int(rand()/float(RAND_MAX+1)*float(target));
}
float Random(float min,float max)
{
    return rand()/float(RAND_MAX+1)*(max-min)+min;
}

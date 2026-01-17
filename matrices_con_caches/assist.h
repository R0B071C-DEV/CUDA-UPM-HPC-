
    int
InitMatrix (
    float * matrix,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT         // matrix height
    )
{
    unsigned int i = 0, j = 0;

    for (i = 0; i < M_HEIGHT; i++) {
      for (j = 0; j < M_WIDTH; j++) {
        matrix[i*M_WIDTH + j] = floorf(100*(rand()/(float)RAND_MAX));
      }
    }
    return (1);
}

    int
GenMatrix (
    float * matrix,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT         // matrix height
    )
{
    unsigned int i = 0, j = 0;

    for (i = 0; i < M_HEIGHT; i++) {
      for (j = 0; j < M_WIDTH; j++) {
        matrix[i*M_WIDTH + j] = i+j+1;
      }
    }
    return (1);
}

    void
CompareMatrix (
    float * const m1,
    float * const m2,
    const unsigned int M_WIDTH,          // matrix width
    const unsigned int M_HEIGHT         // matrix height
    )
{
    unsigned int i = 0, j = 0, wrong = 0;
    int check_ok = 1;

    for (i = 0; i < M_HEIGHT && wrong < 15; i++) {
        for (j = 0; j < M_WIDTH && wrong < 15; j++) {
            if (m1[i*M_WIDTH+j] != m2[i*M_WIDTH+j]) {
                printf ("m1[%d][%d] != m2[%d][%d] : %d != %d\n",
                        i,j,i,j, m1[i*M_WIDTH+j], m2[i*M_WIDTH+j]);
                check_ok = 0; wrong++;
            }
        }
    }
    printf ("    Check ok? ");
    if (check_ok) printf ("Passed.\n");
    else printf ("Failed.\n");
}

    float
CheckSum(const float *matrix, const int width, const int height)
{
    int i, j;
    float s1, s2;

    for (i = 0, s1 = 0; i < width; i++) {
        for (j = 0, s2 = 0; j < height; j++) {
            s2 += matrix[i * width + j];
        }
        s1 += s2;
    }

    return s1;
}



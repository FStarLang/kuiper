#ifndef __TIMING_H
#define __TIMING_H

#include <time.h>
#include <stdio.h>

static inline float __tdiff(struct timespec t1, struct timespec t2)
{
	return (t2.tv_sec - t1.tv_sec)
		+ 1e-9 * (t2.tv_nsec - t1.tv_nsec);
}

/* Set to 0 to not print the parallel factor */
static int time_par = 0;

/* Set to 0 to disable all messages. */
static int verbose = 0;

#define __TIME(s, expr, v) ({						\
		struct timespec __t1p, __t2p;				\
		struct timespec __t1w, __t2w;				\
		typeof(expr) ret;					\
		float *__vv = v;					\
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &__t1p);	\
		clock_gettime(CLOCK_REALTIME, &__t1w);			\
		verbose && fprintf(stderr, "About to call `" s "` \n");	\
		ret = expr;						\
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &__t2p);	\
		clock_gettime(CLOCK_REALTIME, &__t2w);			\
		verbose &&						\
		  fprintf(stderr, "computed `" s "` in %.5fs total "	\
				"CPU time (wall time = %.5fs)\n",	\
				__tdiff(__t1p, __t2p),			\
				__tdiff(__t1w, __t2w));			\
		if (verbose && time_par)				\
			fprintf(stderr, "(parallel factor = %.2f)\n",	\
					__tdiff(__t1p, __t2p)/		\
					__tdiff(__t1w, __t2w));		\
		if (__vv)						\
		        *__vv = __tdiff(__t1w, __t2w);			\
		ret;})

/*
 * Dada una expresión [expr] y un puntero a float [v],
 * imprime el tiempo que tardó en computar la expresión
 * y lo guarda en v (si v != NULL).
 */
#define TIME0(expr, v) __TIME(#expr, expr, v)
/* Extra indirection expands stmt before we stringify it. */
#define TIME(expr, v) TIME0(expr, v)

/* Similar pero para statements. */
#define TIME_void0(stmt, v) __TIME(#stmt, (stmt, 1), v)
/* Extra indirection expands stmt before we stringify it. */
#define TIME_void(stmt, v) TIME_void0(stmt, v)

#define __TIMEREP(s, n, expr, v)		\
	__TIME(s,				\
		({				\
		int __i;			\
		int __n = n;			\
		typeof(expr) __r;		\
		for (__i = 0; __i < __n; ++__i)	\
			__r = expr;		\
		__r; }), v )

/*
 * Similar a TIME pero repite la computación [n] veces.
 * El tiempo *total* se guarda en [v].
 */
#define TIMEREP(n, expr, v) __TIMEREP(#expr, n, expr, v)

/* Similar a TIMEREP, para statements. */
#define TIMEREP_void(n, stmt, v) __TIMEREP(#stmt, n, (stmt, 1), v)

// Find another home
#define __TOKENPASTE(x, y) x ## y
#define TOKENPASTE(x, y) __TOKENPASTE(x, y)
#define TOKENPASTE3(x, y,z) TOKENPASTE(TOKENPASTE(x, y), z)
#define TOKENPASTE5(v, w, x, y, z) TOKENPASTE(TOKENPASTE(TOKENPASTE(TOKENPASTE(v, w), x), y), z)
#define TOKENPASTE6(u, v, w, x, y, z) TOKENPASTE(TOKENPASTE5(u, v, w, x, y), z)

#endif

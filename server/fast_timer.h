#ifndef SPLOTCH_UTILS_FASTTIMER_H
#define SPLOTCH_UTILS_FASTTIMER_H

class fast_timer
{
public:
	void resize(unsigned i)
	{
		t0.resize(i);
		td.resize(i);
		size = i;
	}

	void start(int i)
	{
		if(i>=size) {
			t0.resize(i+1);
			td.resize(i+1);
			size = i+1;			
		}
		t0[i] =  std::chrono::high_resolution_clock::now();
		td[i] = t0[i];
	}


	long mark(int i, std::string msg)
	{
		if(i>=size) return -1;
		tp = std::chrono::high_resolution_clock::now();
		auto count = std::chrono::duration_cast<std::chrono::milliseconds>(tp - td[i]).count();
		std::cout << msg <<  count << "\n";
	    td[i] = tp;
	    return (long)count;
	}

	long mark(int i)
	{
		if(i>=size) return -1;
		tp = std::chrono::high_resolution_clock::now();
		auto count = std::chrono::duration_cast<std::chrono::milliseconds>(tp - td[i]).count();
	    td[i] = tp;
	    return (long)count;		
	}

private:
	std::vector<std::chrono::high_resolution_clock::time_point> t0;
	std::vector<std::chrono::high_resolution_clock::time_point> td;
	std::chrono::high_resolution_clock::time_point tp;
	unsigned size = 0;
};

#endif

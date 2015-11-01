/*
 * Copyright (c) 2004-2014
 *              Tim Dykes University of Portsmouth
 *              Claudio Gheller ETH-CSCS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */


#ifndef MIC_TIMER_H
#define MIC_TIMER_H

#include <vector>
#include <string>
#include <utility>
#include <sys/time.h>

class Timer
{
public: 
	Timer() {};
	~Timer() {};

	void reserve(int i)
	{
		records.reserve(i);
	}

	void start(std::string key)
	{
		for(unsigned i = 0; i < records.size(); i++)
		{
			if(records[i].first == key)
			{
				records[i].second -= (getTime()); 
				return;
			}
		}
		records.push_back(make_pair(key, 0.0));
		records.back().second = -(getTime());
	}

	void stop(std::string key)
	{
		double t = getTime();
		for(unsigned i = 0; i < records.size(); i++)
		{
			if(records[i].first == key) 
			{
				records[i].second += t;
				return;
			}
		}
		printf("Timer.stop(): key: %s not found in list.\n",key.c_str());
		fflush(0);
	}

	void report()
	{
		printf("Timer Report:\n\tKey\t\t\t\tTime\n");
		for(unsigned i = 0; i < records.size(); i++)
		{
			printf("\t%s\t\t\t%f\n",records[i].first.c_str(), records[i].second);
		}
		fflush(0);
	}

	void clear()
	{
		records.clear();
	}

private:
	double getTime()
	{
	    struct timeval t;
	    gettimeofday(&t, NULL);
	    return t.tv_sec + 1e-6*t.tv_usec;
	}

	std::vector<std::pair<std::string, double> > records;


};


#endif
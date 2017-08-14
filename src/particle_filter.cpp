/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	cout << x <<","<<y<<","<<theta<<","<<std[0] <<" " << std[1] <<" " << std[2] << endl;
	num_particles = 1000;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	weights.resize(num_particles);
	for (int i=0; i<num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
		weights[i] = 1.;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	for (int i=0; i<num_particles; i++) {
		double x_f = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t) - sin(particles[i].theta));
		double y_f = particles[i].y + velocity/yaw_rate*(-cos(particles[i].theta+yaw_rate*delta_t) + cos(particles[i].theta));
		double theta_f = particles[i].theta + yaw_rate*delta_t;

		default_random_engine gen;
		normal_distribution<double> dist_x(x_f, std_pos[0]);
		normal_distribution<double> dist_y(y_f, std_pos[1]);
		normal_distribution<double> dist_theta(theta_f, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(const std::vector<Map::single_landmark_s> &map, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int o=0; o<observations.size(); o++) {
		LandmarkObs *obs = &observations[o];
		double min = DBL_MAX;
		for (int m=0; m<map.size(); m++) {
			double dist_t = dist(map[m].x_f,map[m].y_f,obs->x,obs->y);
			if (dist_t < min) {
				obs->id = m;
			}
		}
	}
}

// convert obs coordinates to map 
void ParticleFilter::coordToMap(std::vector<LandmarkObs> &obs, const Particle &p)
{
	for (int i=0; i<obs.size(); i++) {
		double x = obs[i].x;
		double y = obs[i].y;
		double c = cos(p.theta);
		double s = sin(p.theta);

		x = x*c - y*s + p.x;
		y = y*c + x*s + p.y;
		obs[i].x = x;
		obs[i].y = y;
	}
}

static double gausian2(double ux, double uy, double stdx, double stdy, double x, double y)
{
	double pi2 = 1./(2*M_PI*stdx*stdy);
	double tx = pow(x-ux,2)/2./pow(stdx,2);
	double ty = pow(y-uy,2)/2./pow(stdy,2);

	return pi2*exp(-tx-ty);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	vector<Map::single_landmark_s> map = map_landmarks.landmark_list;

	for (int i=0; i<num_particles; i++) {
		vector<LandmarkObs> p_obs = observations;
		coordToMap(p_obs, particles[i]);
		dataAssociation(map, p_obs);
		double w = 1.;
		for (int o=0; o<p_obs.size(); o++) {
			double ux = map[p_obs[o].id].x_f;
			double uy = map[p_obs[o].id].y_f;
			double x = p_obs[o].x;
			double y = p_obs[o].y;
			if (dist(ux,uy,particles[i].x,particles[i].y) > sensor_range) {
				w = 0.;
				break;
			}
			w *= gausian2(ux,uy,std_landmark[0],std_landmark[1],x,y);
		}
		particles[i].weight = w;
		weights[i] = w;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> samples(num_particles);
	default_random_engine gen;
	discrete_distribution<int> dis(weights.begin(),weights.end());
	for (int i=0; i<num_particles; i++) {
		int idx = dis(gen);
		samples.push_back(particles[idx]);
	}
	particles = samples;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

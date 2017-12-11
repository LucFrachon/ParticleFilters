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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 10;

  
  for(int i = 0; i < num_particles; i++)
  {
    Particle p;
    p.id = i;
    p.x = sampleFromUnivariateNormal(x, std[0]);  //sample initial value of x
    p.y = sampleFromUnivariateNormal(y, std[1]);  //sample initial value of y
    p.theta = sampleFromUnivariateNormal(theta, std[2]);  //sample initial value of theta
    p.weight = 1.0;  //initial weight of all particles is 1.0

    particles.push_back(p);  //add particle to particle filter
    weights.push_back(1.0);  //set all initial weights to 1.0
  }

  /* === Uncomment for debugging ===
  cout << "Filter initialized, initial positions:\n";
  for(int i = 0; i < num_particles; i++)
  {
    printParticle(particles[i]);
  }
  */
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  //for each particle:
  for(int i = 0; i < num_particles; i++)
  {
    //define useful values for convenience
    double xi = particles[i].x;
    double xf;
    double yi = particles[i].y;
    double yf;
    double theta_i = particles[i].theta;
    double theta_f;

    if(yaw_rate > 0.0001)
    {
      //compute the projected value of xf, yf, thetaf using control inputs when yaw_rate != 0 
      xf = xi + velocity / yaw_rate * (sin(theta_i +  yaw_rate * delta_t) - sin(theta_i));
      yf = yi + velocity / yaw_rate * (-cos(theta_i + yaw_rate * delta_t) + cos(theta_i));
      theta_f = theta_i + yaw_rate * delta_t;

    } else
    {
      //or compute the projected value of xf, yf, thetaf using control inputs when yaw_rate == 0
      xf = xi + velocity * delta_t * cos(theta_i);
      yf = yi + velocity * delta_t * sin(theta_i);
      theta_f = theta_i;
    }

    //sample a value for x, y, theta from a normal distribution with mean = xf, yf, thetaf
    particles[i].x = sampleFromUnivariateNormal(xf, std_pos[0]);
    particles[i].y = sampleFromUnivariateNormal(yf, std_pos[1]);
    particles[i].theta = sampleFromUnivariateNormal(theta_f, std_pos[2]);
  }
  
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the association's x mapping already converted to world coordinates
    // sense_y: the association's y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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

float ParticleFilter::sampleFromUnivariateNormal(float mean, float stdev)
{
  normal_distribution<double> dist(mean, stdev);
  return dist(gen);
}

void ParticleFilter::printParticle(const Particle p)
{
  cout << "\nParticle id:\t" << p.id << endl
       << "Position x:\t" << p.x << endl
       << "Position y:\t" << p.y << endl
       << "Bearing theta:\t" << p.theta << endl
       << "Weight:\t" << p.weight << endl;

}
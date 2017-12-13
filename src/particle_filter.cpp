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

  num_particles = 50;

  
  for(int i = 0; i < num_particles; ++i)
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

  cout << "Filter initialized, initial positions:\n";
  cout << "GPS Position:\n" 
       << x << endl
       << y << endl
       << theta << endl;

  /*
  for(int i = 0; i < num_particles; ++i)
  {
    printParticle(particles[i]);
  }
  */

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  //for each particle:
  for(int i = 0; i < num_particles; ++i)
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
    particles[i].theta = sampleFromUnivariateNormal(theta_f, 2 * std_pos[2]);

    /*
    cout << "\nPositions after prediction step:\n ";
    printParticle(particles[i]);
    */
  }
  
}

inline LandmarkObs ParticleFilter::transformObservation(LandmarkObs& observation, Particle p)
{
  LandmarkObs transformed_obs = observation;

  //readability
  double x_in = observation.x;
  double y_in = observation.y;
  double x_p = p.x;
  double y_p = p.y;
  double t_p = p.theta;

  //perform conversion operations (rotation + translation, no scaling nor shearing)
  double x_out = x_in * cos(t_p) - y_in * sin(t_p) + x_p;
  double y_out = x_in * sin(t_p) + y_in * cos(t_p) + y_p;

  //overwrite initial values
  observation.x = x_out;
  observation.y = y_out; 
}


void ParticleFilter::transformAndAssociate(Particle& particle, 
    std::vector<LandmarkObs>& observations, const double sensor_range, const Map &map, double sigma_pos[]) {
  //for a single particle:
  
  //cout << "\n\nParticle " << particle.id << endl;
  //cout << "\nNumber of observations: " << observations.size() << endl;

  for(int obs = 0; obs < observations.size(); ++obs)
  {
    /*
    cout << "\nInitial coordinates:\t"
         << "\t" << observations[obs].x
         << "\t" << observations[obs].y
         << endl;
    */

    //convert each observation to map coordinates using particle's position
    transformObservation(observations[obs], particle);

    /*
    cout << "Transformed coordinates:\t"
         << "\t" << observations[obs].x
         << "\t" << observations[obs].y
         << endl;
    */

    //initialize minimum distance and its index
    double min_distance = 999999;
    int nearest_landmark_index = -1;

    //associate observation (now in vehicle coordinates) with nearest neighbour landmark
    for(int lm = 0; lm < map.landmark_list.size(); ++lm)
    {
      double xl = map.landmark_list[lm].x_f;
      double yl = map.landmark_list[lm].y_f;
      double landmark_to_particle = dist(particle.x, xl, particle.y, yl);
      
      //ignore any landmarks whose distance to the particle is higher than the sensor range
      //if(landmark_to_particle > sensor_range) continue;

      double xo = observations[obs].x;
      double yo = observations[obs].y;
      double distance = dist(xo, yo, xl, yl);  //compute Euclidian distance to landmark

      //keep track of the shortest distance and the index of the corresponding landmark
      if(distance < min_distance)
      {
        nearest_landmark_index = lm;
        min_distance = distance;
      }
    }
    
    //cout << "Nearest landmark index and distance:\t" << nearest_landmark_index << "\t" << min_distance << endl;
    
    //add the nearest landmark index to the particle's association list
    particle.associations.push_back(nearest_landmark_index);
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, const Map &map) {

  //reinitialize list of particle weights
  weights.clear();

  for(int i = 0; i < num_particles; ++i)
  {
    //initialize a copy of observations which we'll use to store transformed observations
    std::vector<LandmarkObs> observations_map = observations;

    particles[i].associations.clear();
    transformAndAssociate(particles[i], observations_map, sensor_range, map, std_landmark);

    /*
    cout << "Associations for particle\t" << i << ":\n";
    for(int j = 0; j < particles[i].associations.size(); ++j)
    {
      cout << "obs " << j << ":\t" << particles[i].associations[j] << endl;
    }
    */

    //compute the probability of each observation according to a multivariate normal distribution

    double joint_prob = 1.;

    for(int obs = 0; obs < observations_map.size(); obs++)
    {
      int landmark_index = particles[i].associations[obs];
      
      //ignore observation if not matched to any landmark
      if(-1 == landmark_index) continue;

      double xo = observations_map[obs].x;
      double yo = observations_map[obs].y;
      double xl = map.landmark_list[landmark_index].x_f;
      double yl = map.landmark_list[landmark_index].y_f;
        
      double obs_prob = computeBivariateNormalProbability(xo, yo, xl, yl, std_landmark[0], std_landmark[1]);
      joint_prob *= obs_prob;

      /*cout << "Transformed observation coordinates used in prob calculation:\t"
           << "\t" << observations_map[obs].x
           << "\t" << observations_map[obs].y
           << endl;
      */

      //cout << "Observation " << obs << " probability =\t" << obs_prob << endl;
    }

    particles[i].weight = joint_prob;
    weights.push_back(joint_prob);

    /*cout << "\nParticle after weight update:\t";
    printParticle(particles[i]);
    */
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight. 

  cout << "\nWeights after update:\n";
  double weight_sum = 0;
  for (int i = 0; i < weights.size(); ++i)
  {
    cout << weights[i] << endl;
    weight_sum += weights[i];
  } 
  cout << "average w " << weight_sum / num_particles << endl;

  std::vector<Particle> new_particles;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  //discrete_distribution already normalizes probabilities

  for (int i = 0; i < num_particles; ++i)
  {
    new_particles.push_back(particles[d(gen)]);
  }
  //replace old particles with new sample
  particles = std::move(new_particles);

  cout << "\nWeights after resampling:\n";
  for (int i = 0; i < weights.size(); ++i) cout << weights[i] << endl;
  
}

Particle ParticleFilter::setAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark index that goes along with each listed association
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
    copy( v.begin(), v.end(), ostream_iterator<double>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<double>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

double ParticleFilter::sampleFromUnivariateNormal(double mean, double stdev)
{
  normal_distribution<double> dist(mean, stdev);
  return dist(gen);
}

double ParticleFilter::computeBivariateNormalProbability(double x, double y, double mu_x, double mu_y, 
  double std_x, double std_y)
{
  return 1. / (2 * M_PI * std_x * std_y) * exp(-(pow(x - mu_x, 2) / (2 * pow(std_x, 2)) + pow(y - mu_y, 2) / (2 * pow(std_y, 2))));
}


void ParticleFilter::printParticle(const Particle p)
{
  cout << "\nParticle id:\t" << p.id << endl
       << "Position x:\t" << p.x << endl
       << "Position y:\t" << p.y << endl
       << "Bearing theta:\t" << p.theta << endl
       << "Weight:\t" << p.weight << endl;

}

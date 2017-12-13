 /*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"
#include <random>

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};



class ParticleFilter {
	
	// Number of particles to draw
	int num_particles; 
	
	// Flag, if filter is initialized
	bool is_initialized;
	
	// Vector of weights of all particles
	std::vector<double> weights;

  // Initialize a random generator
  std::default_random_engine gen;

  /*
   * transformObservation: For a single particle, converts a single observation from vehicle to map
   *   coordinates.
   * @param particle: The particle in question, in relation to which we convert observation coordinates.
   * @param observations: The list of observations received from sensors. This will be converted to map
   *   coordinates and the converted coordinates will replace the initial elements of `observations`.
   */
  inline LandmarkObs transformObservation(LandmarkObs& observation, Particle p);

	
public:
	
	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param num_particles Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	/**
	 * transformAndAssociate Tranforms the observation from vehicle to map coordinates, then 
   *   finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association), for a single particle.
	 * @param predicted Vector of predicted landmark observations, in vehicle coordinates
	 * @param observations Vector of landmark observations, vehicle coordinates
   * @map_landmarks A Map object that contains actual landmark positions in map coordinates 
   */
void transformAndAssociate(Particle& particle, 
    std::vector<LandmarkObs>& observations, const double sensor_range, const Map &map_landmarks, 
    double sigma_pos[]);
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations,
			const Map &map);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle setAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y);

	
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized;
	}

  /*
  Sample a scalar value from a univariate normal distribution.
  */    
  double sampleFromUnivariateNormal(double mean, double stdev);

  /*
  Compute probability from a bivariate normal distribution, given two values x and y,
  two means mu_x and mu_y and two standard deviation values std_x and std_y.
  */    
  double computeBivariateNormalProbability(double x, double y, double mu_x, double mu_y, 
    double std_x, double std_y);


  /*
  Debugging function that displays coordinates of a particle and its associated weight
  */
  void printParticle(const Particle p);

};



#endif /* PARTICLE_FILTER_H_ */

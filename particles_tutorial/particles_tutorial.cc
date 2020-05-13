
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/generators.h>
#include <deal.II/particles/particle_handler.h>

#include <fstream>
#include <tuple>

using namespace dealii;


// Creating the function for the velocity profile.
template <int dim>
class SingleVortex : public Function<dim>
{
public:
  SingleVortex()
    : Function<dim>(dim)
  {}
  virtual void
  vector_value(const Point<dim> &point, Vector<double> &values) const override;
};

template <int dim>
void
SingleVortex<dim>::vector_value(const Point<dim> &point,
                                Vector<double> &  values) const
{
  const double T = 2;
  // Bruno
  // So what happens is that the Functions base class contains the time
  // However it is a private member
  // So you need to do this:
  // 1. this-> allows you access functions of the base class of class
  // SingleVortex (so Functions)
  // 2. You get the time using the get_time()
  const double t = this->get_time();

  // Bruno
  // It is ok to have these intermediary variables :)
  // but it does not make anything faster here :)
  const double px = numbers::PI * point(0);
  const double py = numbers::PI * point(1);
  const double pt = numbers::PI / T * t;

  if (dim == 2)
    {
      values[0] = -2 * cos(pt) * pow(sin(px), 2) * sin(py) * cos(py);
      values[1] = 2 * cos(pt) * pow(sin(py), 2) * sin(px) * cos(px);
    }
  else if (dim == 3)
    {
      values[0] = -2 * cos(pt) * pow(sin(px), 2) * sin(py) * cos(py);
      values[1] = 2 * cos(pt) * pow(sin(py), 2) * sin(px) * cos(px);
      values[2] = 0;
    }
}


// Solver
template <int dim>
class moving_particles
{
public:
  moving_particles();
  void
  run();

private:
  void
  particles_generation();
  void
  parallel_weight();
  void
  setup_background_dofs();
  void
  interpolate();
  void
  euler(double dt);
  void
  field_euler(double t, double dt, double T);
  void
  RK4(double t, double dt, double T);
  void
  field_RK4(double t, double dt, double T);
  void
  output_particles(int it, int outputFrequency);
  void
  output_background(int it, int outputFrequency);
  void
  error_estimation();

  MPI_Comm                                  mpi_communicator;
  parallel::distributed::Triangulation<dim> background_triangulation;
  parallel::distributed::Triangulation<dim> particle_triangulation;
  MappingQ<dim>                             mapping;
  Particles::ParticleHandler<dim>           particle_handler;

  // Bruno
  // This FE is only used to generate the particles. It would be a better idea
  // to create it locally instead of keeping as a class member
  FE_Q<dim>                particles_fe;
  DoFHandler<dim>          particles_dof_handler;
  Particles::Particle<dim> particles;

  DoFHandler<dim> background_dof_handler;
  FESystem<dim>   background_fe;


  // Bruno
  // Look at step-40 to see how MPI vectors work
  // You need an "owned" and a  "relevant" copy
  // and they can be both manipulated differently
  TrilinosWrappers::MPI::Vector field_owned;
  TrilinosWrappers::MPI::Vector field_relevant;



  // Bruno
  // Would be better to replace that with an std::vector<Point<dim> > because
  // this is in fact a vector of points ;) What is difficult about this in
  // parallel is that we will need to find the id, like a particle could move
  // from one processor to another. I still have to think about this part ;)
  double **initial_position;

  // Bruno
  // Time of the function will be modified so we need to keep it as a class
  // member. See above and below
  SingleVortex<dim> velocity;

  ConditionalOStream pcout;
};

template <int dim>
moving_particles<dim>::moving_particles()
  : mpi_communicator(MPI_COMM_WORLD)
  , background_triangulation(MPI_COMM_WORLD)
  , particle_triangulation(MPI_COMM_WORLD)
  , mapping(3)
  , particle_handler(particle_triangulation, mapping)
  , particles_fe(1)
  , particles_dof_handler(particle_triangulation)
  , background_dof_handler(background_triangulation)
  , background_fe(FE_Q<dim>(1), dim)
  , pcout({std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0})

{}


// Generation of particles using the grid where particles are generated at the
// locations of the degrees of freedom.
template <int dim>
void
moving_particles<dim>::particles_generation()
{
  // Create a square triangulation
  GridGenerator::hyper_cube(background_triangulation, 0, 1);
  background_triangulation.refine_global(4);
  // Establish where the particles are living
  particle_handler.initialize(background_triangulation, mapping);

  // Generate the necessary bounding boxes for the generator of the particles
  const auto my_bounding_box = GridTools::compute_mesh_predicate_bounding_box(
    background_triangulation, IteratorFilters::LocallyOwnedCell());
  const auto global_bounding_boxes =
    Utilities::MPI::all_gather(MPI_COMM_WORLD, my_bounding_box);

  Point<dim> center;
  if (dim == 2)
    {
      center[0] = 0.5;
      center[1] = 0.75;
    }
  else if (dim == 3)
    {
      center[0] = 0.5;
      center[1] = 0.75;
      center[2] = 0.5;
    }

  const double outer_radius = 0.15;
  const double inner_radius = 0.001;

  // Generation and refinement of the grid where the particles will be created.
  GridGenerator::hyper_shell(
    particle_triangulation, center, inner_radius, outer_radius, 6);
  particle_triangulation.refine_global(3);
  particles_dof_handler.distribute_dofs(particles_fe);

  // Generation of the particles using the Particles::Generators
  Particles::Generators::dof_support_points(particles_dof_handler,
                                            global_bounding_boxes,
                                            particle_handler);


  // Displaying the degrees of freedom
  pcout << "Number of degrees of freedom: " << particles_dof_handler.n_dofs()
        << std::endl;

  // Displaying the total number of generated particles in the domain
  pcout << "Number of Particles: " << particle_handler.n_global_particles()
        << std::endl;

  //  int n = particle_handler.n_global_particles();
  //  initial_position = new double*[n];
  //  int i = 0;

  //  // Get initial location of particles
  //  for (auto particle = particle_handler.begin(); particle !=
  //  particle_handler.end(); ++particle)
  //  {
  //    initial_position[i] = new double[2];
  //    initial_position[i][0] = particle->get_location()[0];
  //    initial_position[i][1] = particle->get_location()[1];

  //    i += 1;
  //   }

  // Outputing the Grid
  std::string grid_file_name("grid-out");
  GridOut     grid_out;
  grid_out.write_mesh_per_processor_as_vtu(background_triangulation,
                                           grid_file_name);
}

// Sets up the background degree of freedom using their interpolation
// And allocated a vector where you can store the entire solution
// of the velocity field
template <int dim>
void
moving_particles<dim>::setup_background_dofs()
{
  background_dof_handler.distribute_dofs(background_fe);
  IndexSet locally_owned_dofs = background_dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(background_dof_handler,
                                          locally_relevant_dofs);

  field_owned.reinit(locally_owned_dofs, mpi_communicator);
  field_relevant.reinit(locally_owned_dofs,
                        locally_relevant_dofs,
                        mpi_communicator);

  pcout << "Number of degrees of freedom in background grid: "
        << background_dof_handler.n_dofs() << std::endl;
}

template <int dim>
void
moving_particles<dim>::interpolate()
{
  // No need for a component mask here since you are interpolating everything
  // The default mask is true to all
  // const ComponentMask & mask = ComponentMask();
  const MappingQ<dim> mapping(1);

  VectorTools::interpolate(
    mapping,
    background_dof_handler,
    velocity,   // third argument is the function you are specifying itself
    field_owned // in this case it is better to store the velocity information
                // in a Trilinos Vector because we will want things to be in
                // parallel
  );
  field_relevant = field_owned;
}


template <int dim>
void
moving_particles<dim>::euler(double dt)
{
  // Bruno
  // We presize the vector for the velocity
  Vector<double> particle_velocity(dim);

  // Looping over all particles in the domain using a particle iterator
  for (auto particle = particle_handler.begin();
       particle != particle_handler.end();
       ++particle)
    {
      // Get the velocity using the current location of particle
      velocity.vector_value(particle->get_location(), particle_velocity);

      Point<dim> particle_location = particle->get_location();
      // Updating the position of the particles and Setting the old position
      // equal to the new position of the particle
      particle_location[0] += particle_velocity[0] * dt;
      particle_location[1] += particle_velocity[1] * dt;

      particle->set_location(particle_location);
    }
}
//}


template <int dim>
void
moving_particles<dim>::parallel_weight()
{
  for (const auto &cell : background_triangulation.active_cell_iterators())
    {
    }
}

template <int dim>
void
moving_particles<dim>::error_estimation()
{
  double l2error = 0;
  double error   = 0;

  int      n = particle_handler.n_global_particles();
  double **final_position;
  final_position = new double *[n];
  int i          = 0;

  for (auto particle = particle_handler.begin();
       particle != particle_handler.end();
       ++particle)
    {
      final_position[i]    = new double[2];
      final_position[i][0] = particle->get_location()[0];
      final_position[i][1] = particle->get_location()[1];

      l2error += (pow((final_position[i][0] - initial_position[i][0]), 2) +
                  pow((final_position[i][1] - initial_position[i][1]), 2));

      i += 1;
    }
  error = sqrt(l2error / (n));
  pcout << "Error " << error << std::endl;
}

template <int dim>
void
moving_particles<dim>::output_particles(int it, int outputFrequency)
{
  // Outputting the results of the simulation at a certain output frequency rate
  if ((it % outputFrequency) == 0)
    {
      Particles::DataOut<dim, dim> particle_output;
      particle_output.build_patches(particle_handler);
      std::string output_folder("output/");
      std::string file_name("particles");

      particle_output.write_vtu_with_pvtu_record(
        output_folder, file_name, it, mpi_communicator, 6);
    }
}


template <int dim>
void
moving_particles<dim>::output_background(int it, int outputFrequency)
{
  // Outputting the results of the simulation at a certain output frequency rate
  if ((it % outputFrequency) == 0)
    {
      std::vector<std::string> solution_names(dim, "velocity");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);


      DataOut<dim> data_out;

      // Attach the solution data to data_out object
      data_out.attach_dof_handler(background_dof_handler);
      data_out.add_data_vector(field_relevant,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);
      Vector<float> subdomain(background_triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = background_triangulation.locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain");

      data_out.build_patches(mapping);

      std::string output_folder("output/");
      std::string file_name("background");

      data_out.write_vtu_with_pvtu_record(
        output_folder, file_name, it, mpi_communicator, 6);
    }
}


// Bruno
// It will be interesting to have two run function. Run with analytical function
// and other one that runs with the interpolation :)
template <int dim>
void
moving_particles<dim>::run()
{
  int    it              = 0;
  int    outputFrequency = 20;
  double t               = 0;
  double T               = 2;
  double dt              = 0.001;

  particles_generation();
  setup_background_dofs();
  interpolate();

  output_particles(it, outputFrequency);
  output_background(it, outputFrequency);


  // Looping over time in order to move the particles using the stream function
  while (t < T)
    {
      // Bruno
      // You can set the time in the class that inherit from function
      // This way your function directly contains the time ;)!
      velocity.set_time(t);
      interpolate();
      euler(dt);
      t += dt;
      ++it;
      particle_handler.sort_particles_into_subdomains_and_cells();
      output_particles(it, outputFrequency);
      output_background(it, outputFrequency);
    }
  //  error_estimation();
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  moving_particles<2>              solution;
  solution.run();

  return 0;
}

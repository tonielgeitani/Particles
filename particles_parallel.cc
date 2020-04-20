
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/particles/generators.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/data_out.h>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <fstream>
#include <tuple>

using namespace dealii;

int argc;
char **argv;

Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
MPI_Comm mpi_communicator(MPI_COMM_WORLD);
const unsigned int n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator));
const unsigned int this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator));
ConditionalOStream pcout(std::cout, (this_mpi_process == 0));

//Solver
template <int dim> class moving_particles
{
  public:
    moving_particles();
    void run(int i, int j);

  private:
    void particles_generation();
    void setup_parallel_vectors();
    void vortex_euler(double t, double dt, double T);
    void field_euler(double t, double dt, double T);
    void vortex_RK4(double t, double dt, double T);
    void field_RK4(double t, double dt, double T);
    void output_results(int it, int outputFrequency);
    void error_estimation();

    //MPI_Comm mpi_communicator;

    //const unsigned int n_mpi_processes;
    //const unsigned int this_mpi_process;
    //ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> particle_triangulation;
    MappingQ<dim> mapping;
    Particles::ParticleHandler<dim> particle_handler;
    FE_Q<dim> particles_fe;
    DoFHandler<dim> particles_dof_handler;
    Particles::Particle<dim> particles;

    TrilinosWrappers::MPI::Vector vx;
    TrilinosWrappers::MPI::Vector vy;
    TrilinosWrappers::MPI::Vector x;
    TrilinosWrappers::MPI::Vector y;

    IndexSet locally_owned_dofs;

    double **initial_position;
};

template <int dim>
moving_particles<dim>::moving_particles()
    : //mpi_communicator(MPI_COMM_WORLD)
    //, n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    //, this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    //, pcout(std::cout, (this_mpi_process == 0))
     particle_triangulation(MPI_COMM_WORLD)
    , mapping(3),particle_handler(particle_triangulation,mapping)
    , particles_fe(1),particles_dof_handler(particle_triangulation)
    {}

auto parameters()
{
  int i, j;
  pcout << "Please insert 0 for Single Vortex or 1 for Deformation Field: " << std::endl;
  std::cin >> i;
  pcout << "Please insert 0 for Euler Solver or 1 for Runge Kutta 4 Solver: " << std::endl;
  std::cin >> j;

  if (i == 0)
  {
    pcout << "The stream function simulated is a Single Vortex. " << std::endl;
  }
  else if (i == 1)
  {
    pcout<< "The stream function simulated is a Deformation Field. " << std::endl;
  }

  if (j == 0)
  {
    pcout << "The solver used is Euler Solver. " << std::endl;
  }
  else if (j == 1)
  {
    pcout << "The solver used is Runge Kutta 4 Solver " << std::endl;
  }

  struct result {int i; int j;};
  return   result {i,j};
}

//Generation of particles using the grid where particles are generated at the
//locations of the degrees of freedom.
template <int dim> void moving_particles<dim>::particles_generation()
{
  Point<dim> 	center;

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

  const double 	outer_radius = 0.15;
  const double 	inner_radius = 0.001;

  //Generation and refinement of the grid where the particles will be created.
  GridGenerator::hyper_shell(particle_triangulation, center, inner_radius, outer_radius, 6);
  particle_triangulation.refine_global(3);
  particles_dof_handler.distribute_dofs(particles_fe);

  // Generate the necessary bounding boxes for the generator of the particles
  const auto my_bounding_box = GridTools::compute_mesh_predicate_bounding_box(particle_triangulation, IteratorFilters::LocallyOwnedCell());
  const auto global_bounding_boxes = Utilities::MPI::all_gather(MPI_COMM_WORLD, my_bounding_box);

  //Generation of the particles using the Particles::Generators
  Particles::Generators::dof_support_points(particles_dof_handler, global_bounding_boxes, particle_handler);

  // Displaying the degrees of freedom
  pcout << "Number of degrees of freedom: " << particles_dof_handler.n_dofs()
            << std::endl;

  // Displaying the total number of generated particles in the domain
  pcout << "Number of Particles: " << particle_handler.n_global_particles()
            << std::endl;

  int n = particle_handler.n_global_particles();
  initial_position = new double*[n];
  int i = 0;

  // Get initial location of particles
  for (auto particle = particle_handler.begin(); particle != particle_handler.end(); ++particle)
  {
    initial_position[i] = new double[2];
    initial_position[i][0] = particle->get_location()[0];
    initial_position[i][1] = particle->get_location()[1];

    i += 1;
   }

  //Outpuytting the Grid
  std::ofstream out("grid.vtu");
  GridOut       grid_out;
  grid_out.write_vtu(particle_triangulation, out);
  pcout << "Grid written to grid.vtu" << std::endl;

  //Displaying the particle distribution before implementation of the velocity profile
  Particles::DataOut<dim,dim> particle_output;
  particle_output.build_patches(particle_handler);
  std::ofstream output("solution-0.vtu");
  particle_output.write_vtu(output);

}

template <int dim> void moving_particles<dim>::setup_parallel_vectors()
{
  locally_owned_dofs = particles_dof_handler.locally_owned_dofs();

  vx.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  vy.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  x.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  y.reinit(locally_owned_dofs, MPI_COMM_WORLD);
}

template <int dim> void moving_particles<dim>::vortex_euler(double t, double dt, double T)
{
  Point<dim> particle_location;

//  for (const auto &cell : particles_dof_handler.active_cell_iterators())
//    if (cell->is_locally_owned())
//    {
      // Looping over all particles in the domain using a particle iterator
      for (auto particle = particle_handler.begin(); particle != particle_handler.end(); ++particle)
      {
        // Get the position of the particle
        double x = particle->get_location()[0];
        double y = particle->get_location()[1];

        // Calculation of the 2 dimensional velocity (single vortex)
        double vx = -2*cos((M_PI/T)*t)*pow(sin(M_PI*x),2)
                    *sin(M_PI*y)*cos(M_PI*y);
        double vy = 2*cos((M_PI/T)*t)*pow(sin(M_PI*y),2)
                    *sin(M_PI*x)*cos(M_PI*x);

        // Updating the position of the particles
        x = x + vx*dt;
        y = y + vy*dt;

        // Setting the old position equal to the new position of the particle
        particle_location[0] = x;
        particle_location[1] = y;

        particle->set_location(particle_location);
      }
//    }
}

template <int dim> void moving_particles<dim>::field_euler(double t, double dt, double T)
{
  Point<dim> particle_location;

    // Looping over all particles in the domain using a particle iterator
    for (auto particle = particle_handler.begin(); particle != particle_handler.end(); ++particle)
    {
        // Get the position of the particle
        double x = particle->get_location()[0];
        double y = particle->get_location()[1];

        // Calculation of the 2 dimensional velocity (deformation field)
        double vx = cos((M_PI/T)*t)*sin(4*M_PI*(x + 0.5))
                    *sin(4*M_PI*(y + 0.5));
        double vy = cos((M_PI/T)*t)*cos(4*M_PI*(x + 0.5))
                    *cos(4*M_PI*(y + 0.5));

        // Updating the position of the particles
        x = x + vx*dt;
        y = y + vy*dt;

        // Setting the old position equal to the new position of the particle
        particle_location[0] = x;
        particle_location[1] = y;

        particle->set_location(particle_location);
    }
}

template <int dim> void moving_particles<dim>::vortex_RK4(double t, double dt, double T)
{
  Point<dim> particle_location;

    // Looping over all particles in the domain using a particle iterator
    for (auto particle = particle_handler.begin(); particle != particle_handler.end(); ++particle)
    {
        // Get the position of the particle
        double x = particle->get_location()[0];
        double y = particle->get_location()[1];

        // Calculation of the 2 dimensional velocity (single vortex)
        double vx = -2*cos((M_PI/T)*t)*pow(sin(M_PI*x),2)
                    *sin(M_PI*y)*cos(M_PI*y);
        double vy = 2*cos((M_PI/T)*t)*pow(sin(M_PI*y),2)
                    *sin(M_PI*x)*cos(M_PI*x);

        // Implementation of the Runge Kutta
        double k1x = vx*dt;
        x = x + (k1x/2);
        double k1y = vy*dt;
        y = y + (k1y/2);

        vx = -2*cos((M_PI/T)*(t+(dt/2)))*pow(sin(M_PI*x),2)
                    *sin(M_PI*y)*cos(M_PI*y);
        vy = 2*cos((M_PI/T)*(t+(dt/2)))*pow(sin(M_PI*y),2)
                    *sin(M_PI*x)*cos(M_PI*x);

        double k2x = vx*dt;
        x = x - (k1x/2) + (k2x/2);
        double k2y = vy*dt;
        y = y - (k1y/2) + (k2y/2);

        vx = -2*cos((M_PI/T)*(t+(dt/2)))*pow(sin(M_PI*x),2)
                    *sin(M_PI*y)*cos(M_PI*y);
        vy = 2*cos((M_PI/T)*(t+(dt/2)))*pow(sin(M_PI*y),2)
                    *sin(M_PI*x)*cos(M_PI*x);

        double k3x = vx*dt;
        x = x  - (k2x/2) + k3x;
        double k3y = vy*dt;
        y = y  - (k2y/2) + k3y;

        vx = -2*cos((M_PI/T)*t)*pow(sin(M_PI*x),2)
                    *sin(M_PI*y)*cos(M_PI*y);
        vy = 2*cos((M_PI/T)*t)*pow(sin(M_PI*y),2)
                    *sin(M_PI*x)*cos(M_PI*x);

        double k4x = vx*dt;
        double k4y = vy*dt;

        // Updating the position of the particles
        x = x - k3x + ((1/6)*(k1x + (2*k2x) + (2*k3x) +k4x));
        y = y - k3y + ((1/6)*(k1y + (2*k2y) + (2*k3y) +k4y));

        // Setting the old position equal to the new position of the particle
        particle_location[0] = x;
        particle_location[1] = y;

        particle->set_location(particle_location);
    }
}

template <int dim> void moving_particles<dim>::field_RK4(double t, double dt, double T)
{
  Point<dim> particle_location;

    // Looping over all particles in the domain using a particle iterator
    for (auto particle = particle_handler.begin(); particle != particle_handler.end(); ++particle)
    {
        // Get the position of the particle
        double x = particle->get_location()[0];
        double y = particle->get_location()[1];

        // Calculation of the 2 dimensional velocity (deformation field)
        double vx = cos((M_PI/T)*t)*sin(4*M_PI*(x + 0.5))
                    *sin(4*M_PI*(y+ 0.5));
        double vy = cos((M_PI/T)*t)*cos(4*M_PI*(x + 0.5))
                    *cos(4*M_PI*(y+ 0.5));

        // Implementation of the Runge Kutta
        double k1x = vx*dt;
        x = x + (k1x/2);
        double k1y = vy*dt;
        y = y + (k1y/2);

        vx = cos((M_PI/T)*(t+(dt/2)))*sin(4*M_PI*(x + 0.5))
                    *sin(4*M_PI*(y + 0.5));
        vy = cos((M_PI/T)*(t+(dt/2)))*cos(4*M_PI*(x+ 0.5))
                    *cos(4*M_PI*(y + 0.5));

        double k2x = vx*dt;
        x = x - (k1x/2) + (k2x/2);
        double k2y = vy*dt;
        y = y - (k1y/2) + (k2y/2);

        vx = cos((M_PI/T)*(t+(dt/2)))*sin(4*M_PI*(x + 0.5))
                    *sin(4*M_PI*(y + 0.5));
        vy = cos((M_PI/T)*(t+(dt/2)))*cos(4*M_PI*(x+ 0.5))
                    *cos(4*M_PI*(y + 0.5));

        double k3x = vx*dt;
        x = x  - (k2x/2) + k3x;
        double k3y = vy*dt;
        y = y  - (k2y/2) + k3y;

        vx = cos((M_PI/T)*t)*sin(4*M_PI*(x + 0.5))
                    *sin(4*M_PI*(y + 0.5));
        vy = cos((M_PI/T)*t)*cos(4*M_PI*(x+ 0.5))
                    *cos(4*M_PI*(y + 0.5));

        double k4x = vx*dt;
        double k4y = vy*dt;

        // Updating the position of the particles
        x = x - k3x + ((1/6)*(k1x + (2*k2x) + (2*k3x) +k4x));
        y = y - k3y + ((1/6)*(k1y + (2*k2y) + (2*k3y) +k4y));

        // Setting the old position equal to the new position of the particle
        particle_location[0] = x;
        particle_location[1] = y;

        particle->set_location(particle_location);
    }
}

template <int dim> void moving_particles<dim>::error_estimation()
{
  double l2error=0;
  double error=0;

  int n = particle_handler.n_global_particles();
  double **final_position;
  final_position = new double*[n];
  int i = 0;

  for (auto particle = particle_handler.begin(); particle != particle_handler.end(); ++particle)
  {
    final_position[i] = new double[2];
    final_position[i][0] = particle->get_location()[0];
    final_position[i][1] = particle->get_location()[1];

    l2error += (pow((final_position[i][0]-initial_position[i][0]),2)+ pow((final_position[i][1]-initial_position[i][1]),2));

    i += 1;
  }
    error = sqrt(l2error/(n));
    pcout << "Error " << error << std::endl;
}

template <int dim> void moving_particles<dim>::output_results(int it, int outputFrequency)
{
  // Outputting the results of the simulation at a certain output frequency rate
  if ((it % outputFrequency)==0)
  {
    Particles::DataOut<dim,dim> particle_output;
    particle_output.build_patches(particle_handler);
    std::ofstream output("solution-" + std::to_string(it) + ".vtu");
    particle_output.write_vtu(output);
  }
}

template <int dim> void moving_particles<dim>::run(int i, int j)
{
  int it = 0;
  int outputFrequency = 50;
  double t = 0;
  double T = 2;
  double dt = 0.001;

  particles_generation();

  // Looping over time in order to move the particles using the stream function
  while (t<T)
  {
    if (i == 0 && j == 0)
    {
      vortex_euler(t, dt, T);
    }
    else if (i == 0 && j == 1)
    {
      vortex_RK4(t, dt, T);
    }
    else if (i == 1 && j == 0)
    {
      field_euler(t, dt, T);
    }
    else if (i == 1 && j == 1)
    {
      field_RK4(t, dt, T);
    }
    output_results(it,outputFrequency);
    t += dt;
    ++ it;
  }
  error_estimation();
}

int main()
{
  auto[i,j] = parameters();
  moving_particles<2> solution;
  solution.run(i,j);

  return 0;
}

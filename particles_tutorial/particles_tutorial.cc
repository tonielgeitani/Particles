
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

template <int dim>
class Interpolate : public Function<dim>
{
public:
  Interpolate()
    : Function<dim>(dim)
  {}
  virtual void
  field_on_particles(const DoFHandler<dim> &                 field_dh,
                    const Particles::ParticleHandler<dim> &particle_handler,
                    TrilinosWrappers::MPI::Vector &          field_vector,
                    TrilinosWrappers::MPI::Vector&           interpolated_field
                    ) const;
};

template <int dim>
void
Interpolate<dim>::field_on_particles(const DoFHandler<dim> &                field_dh,
                                    const Particles::ParticleHandler<dim> &particle_handler,
                                    TrilinosWrappers::MPI::Vector &   field_vector,
                                    TrilinosWrappers::MPI::Vector&          interpolated_field
                                    ) const
{
    const ComponentMask &field_comps = ComponentMask();
    if (particle_handler.n_locally_owned_particles() == 0)
             {
               interpolated_field.compress(VectorOperation::add);
               return; // nothing else to do here
             }

           const auto &tria     = field_dh.get_triangulation();
           const auto &fe       = field_dh.get_fe();
           auto        particle = particle_handler.begin();
           const auto  max_particles_per_cell =
             particle_handler.n_global_max_particles_per_cell();

           // Take care of components
           const ComponentMask comps =
             (field_comps.size() == 0 ? ComponentMask(fe.n_components(), true) :
                                        field_comps);
           AssertDimension(comps.size(), fe.n_components());
           const auto n_comps = comps.n_selected_components();

           AssertDimension(field_vector.size(), field_dh.n_dofs());
           AssertDimension(interpolated_field.size(),
                           particle_handler.get_next_free_particle_index() *
                             n_comps);

           // Global to local indices
           std::vector<unsigned int> space_gtl(fe.n_components(),
                                               numbers::invalid_unsigned_int);
           for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
             if (comps[i])
               space_gtl[i] = j++;

           std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

           while (particle != particle_handler.end())
             {
               const auto &cell = particle->get_surrounding_cell(tria);
               const auto &dh_cell =
                 typename DoFHandler<dim>::cell_iterator(*cell, &field_dh);
               dh_cell->get_dof_indices(dof_indices);
               const auto pic         = particle_handler.particles_in_cell(cell);
               const auto n_particles = particle_handler.n_particles_in_cell(cell);

               Assert(pic.begin() == particle, ExcInternalError());
               for (unsigned int i = 0; particle != pic.end(); ++particle, ++i)
                 {
                   const auto &reference_location =
                     particle->get_reference_location();

                   const auto id = particle->get_id();

                   for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                     {
                       const auto comp_j =
                         space_gtl[fe.system_to_component_index(j).first];
                       if (comp_j != numbers::invalid_unsigned_int)
                         interpolated_field[id * n_comps + comp_j] +=
                           fe.shape_value(j, reference_location) *
                           field_vector(dof_indices[j]);
                     }

                  }
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
  interpolate_field();
  void
  interpolate_field_on_particles();
  void
  euler_for_interpolated_values(double dt);
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
  Interpolate<dim> interpolation;

  ConditionalOStream pcout;
};

template <int dim>
moving_particles<dim>::moving_particles()
  : mpi_communicator(MPI_COMM_WORLD)
  , background_triangulation(MPI_COMM_WORLD)
  , particle_triangulation(MPI_COMM_WORLD)
  , mapping(1)
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
moving_particles<dim>::interpolate_field()
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
moving_particles<dim>::interpolate_field_on_particles()
{
    ComponentMask mask(background_fe.n_components(), true);
    const auto    n_comps = mask.n_selected_components();

    const auto n_local_particles_dofs =
         particle_handler.n_locally_owned_particles() * n_comps;

    auto particle_sizes =
         Utilities::MPI::all_gather(MPI_COMM_WORLD, n_local_particles_dofs);

    //const auto my_start = std::accumulate(particle_sizes.begin(),
    //                                         particle_sizes.begin() + my_mpi_id,
    //                                         0u);

    IndexSet local_particle_index_set(particle_handler.n_global_particles() *
                                         n_comps);

    //local_particle_index_set.add_range(my_start,
    //                                      my_start + n_local_particles_dofs);

//    auto global_particles_index_set =
//         Utilities::MPI::all_gather(MPI_COMM_WORLD, n_local_particles_dofs);

    TrilinosWrappers::MPI::Vector interpolation_on_particles(
         local_particle_index_set, MPI_COMM_WORLD);

    interpolation.field_on_particles(
         background_dof_handler, particle_handler, field_relevant, interpolation_on_particles);

}

template <int dim>
void
moving_particles<dim>::euler_for_interpolated_values(double dt)
{

}

//template <int dim>
//void
//moving_particles<dim>::euler(double dt)
//{
//  // Bruno
//  // We presize the vector for the velocity
//  Vector<double> particle_velocity(dim);

//  // Looping over all particles in the domain using a particle iterator
//  for (auto particle = particle_handler.begin();
//       particle != particle_handler.end();
//       ++particle)
//    {
//      // Get the velocity using the current location of particle
//      velocity.vector_value(particle->get_location(), particle_velocity);

//      Point<dim> particle_location = particle->get_location();
//      // Updating the position of the particles and Setting the old position
//      // equal to the new position of the particle
//      particle_location[0] += particle_velocity[0] * dt;
//      particle_location[1] += particle_velocity[1] * dt;

//      particle->set_location(particle_location);
//    }
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
  interpolate_field();

  output_particles(it, outputFrequency);
  output_background(it, outputFrequency);


  // Looping over time in order to move the particles using the stream function
  while (t < T)
    {
      // Bruno
      // You can set the time in the class that inherit from function
      // This way your function directly contains the time ;)!
      velocity.set_time(t);
      interpolate_field();
      interpolate_field_on_particles();
      //euler(dt);
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

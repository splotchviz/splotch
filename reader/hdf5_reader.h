/*
  File reader for HDF5 data
  Implements IFile interface in reader.h
*/
#ifdef HDF5

#include "IFile.h"
#include "hdf5.h"
#include <stddef.h>

class HDF5File : public IFile 
{
 public: 
  // Try to open file
  void open(paramfile& params) override {
    std::string infile = params.find<std::string>("infile");
    // H5Fopen returns negative number for failed open
    file_id = H5Fopen(infile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file_id < 0)
      planck_fail("HDF5 Reader: Could not open file "+infile);

    file_open = true;
    name = infile;
  }

  void close(paramfile& params) override {
    if(file_open){
      H5Fclose(file_id);
      file_open = false;
      file_id = -1;
    }
  }

  // Get file info
  void read_info(paramfile& params) override {
    planck_assert(file_open, "HDF5 Reader: file not open");
    printf("Reading file structure");
    printf("Warning: currently groups are not preserved, and datasets with more than one reference count will be ignored\n");
    auto op_func = [](hid_t loc_id, const char *name, const H5O_info_t *info, void *operator_data){
      // Get our op_data
      std::vector<group>* gs = static_cast<std::vector<group>*>(operator_data);

      // Group identifier
      int gid = 0;
      // For the root group, create the first group
      if (name[0] == '.')
        gs->push_back(group("."));
      else
        switch (info->type) {
          case H5O_TYPE_GROUP:
            printf ("Warning, ignoring group: %sn", name);
            //printf ("%s  (Group)\n", name);
            break;
          case H5O_TYPE_DATASET:
            if(info->rc!=1){
              printf ("Warning, ignoring multiply referenced dataset: %sn", name);
            } else {
              (*gs)[gid].datasets.push_back(dataset(name));
            }
            //printf ("%s  (Dataset)\n", name);  
            break;          
          case H5O_TYPE_NAMED_DATATYPE: 
            printf ("Warning, ignoring named datatype: %sn", name);
             //printf ("%s  (Datatype)\n", name);
            break;
          default:
            printf ("Warning, ignoring unknown: %sn", name);
            //printf ("%s  (Unknown)\n", name);
            break;
          }

      return 0;
    };

    herr_t status = H5Ovisit (file_id, H5_INDEX_NAME, H5_ITER_NATIVE, op_func, &groups);
    if(status)
      planck_fail("HDF5 Reader: H5Ovisit failed");

    info_read = true;
  }

  // Fill point array with splotch formatted particles
  void read(paramfile &params, std::vector<particle_sim> &points) override {
    
    bool was_open = file_open;
    if(!was_open) {
      open(params);
    }
    
    if(!info_read)
      read_info(params);

    // Get the fieldnames 
    fields.resize(0);
    fields.push_back(params.find<std::string>("x",  ""));
    fields.push_back(params.find<std::string>("y",  ""));
    fields.push_back(params.find<std::string>("z",  ""));
    fields.push_back(params.find<std::string>("C1"));
    fields.push_back(params.find<std::string>("C2",  ""));
    fields.push_back(params.find<std::string>("C3",  ""));
    fields.push_back(params.find<std::string>("r",  ""));
    fields.push_back(params.find<std::string>("I",  ""));

    // Get particle_sim struct offsets 
    // Should be same as used in HOFFSET
    std::vector<int> offsets;
    offsets.push_back(offsetof(particle_sim, x));
    offsets.push_back(offsetof(particle_sim, y));
    offsets.push_back(offsetof(particle_sim, z));
    offsets.push_back(offsetof(particle_sim, e.r));
    offsets.push_back(offsetof(particle_sim, e.g));
    offsets.push_back(offsetof(particle_sim, e.b));
    offsets.push_back(offsetof(particle_sim, r));
    offsets.push_back(offsetof(particle_sim, I));

    // Validate the fields
    for(auto f : fields){
      if(f != ""){
        bool found = false;
        for(auto g : groups)
          for(auto d : g.datasets)
            if(d.name==f) found = true;
        if(!found)
          planck_fail("HDF5 Reader: field "+f+" not found");
      }
    }
    // HDF identifiers
    hid_t dataset_space;
    hid_t memory_space;
    hid_t dataset_id;
    hid_t space_dims;


    // The field we will use to determine size of dataset
    // Default to X field
    int range_field = FieldId::F_X;
    // Check for grid data, if so we use C1 field instead of X field
    bool grid = false;
    if(fields[0] == ""){
      range_field = FieldId::F_CR;
      grid=true;
    }

    // Open the ranging dataset and get space
    dataset_id = H5Dopen1(file_id,fields[range_field].c_str());
    dataset_space = H5Dget_space(dataset_id);
    // Get the dimensionality of the space
    // (unintuitively called rank in hdf5, call it space_dims here)
    space_dims = H5Sget_simple_extent_ndims(dataset_space);
    // Get size of each dimension
    std::vector<hsize_t> dim_extents(space_dims);
    std::vector<hsize_t> dim_extents_max(space_dims);
    std::vector<hsize_t> local_grid(space_dims);
    std::vector<hsize_t> start(space_dims);
    H5Sget_simple_extent_dims(dataset_space, &dim_extents[0], &dim_extents_max[0]);
    H5Dclose(dataset_id);

    // For parallel read of grid we can only read if extent in X axis is larger than 
    if(dim_extents[0]<mpiMgr.num_ranks()){
      planck_fail("HDF5 Reader: cannot parallel read, more processors than points \
                  (or for grid data, than extent of X dim)");
    } 

    // Work out parallel read extents for each processor, 
    // either in 1d (where XYZ is provided) or 3D (where XYZ is inferred from grid)
    long np_local, np_total;
    np_total = (long)dim_extents[0];
    long step = np_total/mpiMgr.num_ranks();
    local_grid[0] = step;
    np_local=local_grid[0];
    if(grid){
      if(mpiMgr.rank() == mpiMgr.num_ranks()-1) 
        local_grid[0] = np_total-(local_grid[0]*(mpiMgr.num_ranks()-1));

      if(space_dims == 3)
      {
        local_grid[1] = (int)dim_extents[1];
        local_grid[2] = (int)dim_extents[2];
        np_local = local_grid[0]*local_grid[1]*local_grid[2];
        np_total = dim_extents[0]*dim_extents[1]*dim_extents[2]; 
      }    
      else{
        planck_fail("HDF5 Reader: grid data expected but field is not 3 dimensional");
      }
    }
    // Starting offsets 
    start[0]  = (hsize_t)(step * mpiMgr.rank());
    if(space_dims == 3)
    {
      start[1]  = 0;
      start[2]  = 0; 
    }

    if (mpiMgr.master())
    {
      std::cout << "Read file: " << name << std::endl;
      std::cout << "Local number of points/cells " << np_local << std::endl;
      std::cout << ((grid) ? "XYZ inferred from grid" : "XYZ provided as fields") << std::endl;;
    }
    // Resize splotch particles
    points.resize(np_local);
    printf("Splotch points: %lu", points.size());

    // Read each field assuming float at the moment
    float* buf = new float[np_local];
    for(unsigned fid = 0; fid < fields.size(); fid++){
      if(fields[fid] != ""){
        dataset_id = H5Dopen1(file_id,fields[fid].c_str());
        dataset_space = H5Dget_space(dataset_id);
        H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, &start[0], NULL, &local_grid[0], NULL); 
        memory_space = H5Screate_simple(space_dims, &local_grid[0], &local_grid[0]); 

        H5Dread(dataset_id, H5T_NATIVE_FLOAT, memory_space, dataset_space, H5P_DEFAULT, buf);

        // #pragma omp parallel for
        for(unsigned i = 0; i < np_local; i++) 
          *((float*)(((char*)&points[i])+offsets[fid])) = buf[i];

        H5Sclose(memory_space);
        H5Sclose(dataset_space); 
        H5Dclose(dataset_id);       
      }
    }
    delete[] buf;

    // Fill non-provided fields
    if(grid){
      // Do xyz
    }

    if(fields[FieldId::F_R]==""){
      float r = params.find<float>("smooth_factor", 1.0);
      for(unsigned i = 0; i < np_local; i++) 
        points[i].r = r;
    }

    if(fields[FieldId::F_I]==""){
      float I = params.find<float>("intensity_factor", 1.0);
      for(unsigned j = 0; j < np_local; j++) 
        points[j].I = I;
    }

    // Type
    for(unsigned i = 0; i < np_local; i++)
      points[i].type = 0;

    if(!was_open){
      close(params);
    }

  }

  void read_field(paramfile &params, std::string field_name, std::vector<float>& storage) override {
    bool was_open = file_open;
    if(!was_open) {
      open(params);
    }
    
    if(!info_read)
      read_info(params);

    // Validate the fieldname
      bool found = false;
      for(auto g : groups)
        for(auto d : g.datasets)
          if(d.name==field_name) found = true;
      if(!found)
        planck_fail("HDF5 Reader: field "+field_name+" not found");


    // HDF identifiers
    hid_t dataset_space;
    hid_t memory_space;
    hid_t dataset_id;
    hid_t space_dims;


    // The field we will use to determine size of dataset
    // Default to X field
    // int range_field = FieldId::F_X;
    // // Check for grid data, if so we use C1 field instead of X field
    // bool grid = false;
    // if(fields[0] == ""){
    //   range_field = FieldId::F_CR;
    //   grid=true;
    // }
    // No grid suport as of yet
    bool grid = false;

    // Open the dataset and get space
    dataset_id = H5Dopen1(file_id,field_name.c_str());
    dataset_space = H5Dget_space(dataset_id);
    // Get the dimensionality of the space
    // (unintuitively called rank in hdf5, call it space_dims here)
    space_dims = H5Sget_simple_extent_ndims(dataset_space);
    // Get size of each dimension
    std::vector<hsize_t> dim_extents(space_dims);
    std::vector<hsize_t> dim_extents_max(space_dims);
    std::vector<hsize_t> local_grid(space_dims);
    std::vector<hsize_t> start(space_dims);
    H5Sget_simple_extent_dims(dataset_space, &dim_extents[0], &dim_extents_max[0]);

    // For parallel read of grid we can only read if extent in X axis is larger than 
    if(dim_extents[0]<mpiMgr.num_ranks()){
      planck_fail("HDF5 Reader: cannot parallel read, more processors than points \
                  (or for grid data, than extent of X dim)");
    } 

    // Work out parallel read extents for each processor, 
    // either in 1d (where XYZ is provided) or 3D (where XYZ is inferred from grid)
    long np_local, np_total;
    np_total = (long)dim_extents[0];
    long step = np_total/mpiMgr.num_ranks();
    local_grid[0] = step;
    np_local=local_grid[0];
    if(grid){
      if(mpiMgr.rank() == mpiMgr.num_ranks()-1) 
        local_grid[0] = np_total-(local_grid[0]*(mpiMgr.num_ranks()-1));

      if(space_dims == 3)
      {
        local_grid[1] = (int)dim_extents[1];
        local_grid[2] = (int)dim_extents[2];
        np_local = local_grid[0]*local_grid[1]*local_grid[2];
        np_total = dim_extents[0]*dim_extents[1]*dim_extents[2]; 
      }    
      else{
        planck_fail("HDF5 Reader: grid data expected but field is not 3 dimensional");
      }
    }
    // Starting offsets 
    start[0]  = (hsize_t)(step * mpiMgr.rank());
    if(space_dims == 3)
    {
      start[1]  = 0;
      start[2]  = 0; 
    }

    if (mpiMgr.master())
    {
      std::cout << "Input data file name: " << name << std::endl;
      std::cout << "Reading " << np_local << " elements of field " << field_name << std::endl;
      std::cout << ((grid) ? "XYZ inferred from grid" : "XYZ provided as fields") << std::endl;;
    }
    // Resize splotch particles
    storage.resize(np_local);

    // Read each field assuming float at the moment
    float* buf = storage.data();
    H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, &start[0], NULL, &local_grid[0], NULL); 
    memory_space = H5Screate_simple(space_dims, &local_grid[0], &local_grid[0]); 
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, memory_space, dataset_space, H5P_DEFAULT, buf);

    H5Sclose(memory_space);
    H5Sclose(dataset_space); 
    H5Dclose(dataset_id);       

    if(!was_open){
      close(params);
    } 
  }


  // Get file info
  void write_filtered(paramfile &params, std::string& filename, std::vector<unsigned long long>& idxs) override {
    bool was_open = file_open;
    if(!was_open) {
      open(params);
    }
    
    if(!info_read)
      read_info(params);
    printf("Writing filtered hdf5 file %s\n", filename.c_str());
    printf("Warning: currently groups are not preserved, named datatypes & unknowns are ignored \
            and datasets with more than one reference count are ignored\n");

    // Open new file for writing
    hid_t outfile_id = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t global_size = idxs.size();
    // Start needs to be determined from parallel prefix sum of idx lengths
    hsize_t local_subset_start = 0;
    hsize_t local_subset_length = idxs.size();

    // For each field, create dataspace & dataset then close
    herr_t status;    
    for(auto d: groups[0].datasets){
      // Create simple 1d array dataspace
      hid_t dataspace = H5Screate_simple(1, &global_size, NULL);
      // Create dataset
      hid_t dataset = H5Dcreate(outfile_id, d.name.c_str(), H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      // Close
      H5Dclose(dataset);
      H5Sclose(dataspace);
    }



    std::vector<float> storage;
    bool redo_indexes = true;
    for(auto d: groups[0].datasets){
      // HDF identifiers
      hid_t dataset_space;
      hid_t memory_space;
      hid_t dataset_id;
      hid_t space_dims;

      // The field we will use to determine size of dataset
      // Default to X field
      // int range_field = FieldId::F_X;
      // // Check for grid data, if so we use C1 field instead of X field
      // bool grid = false;
      // if(fields[0] == ""){
      //   range_field = FieldId::F_CR;
      //   grid=true;
      // }
      // No grid suport as of yet
      bool grid = false;

      // Open the dataset and get space
      dataset_id = H5Dopen1(file_id,d.name.c_str());
      dataset_space = H5Dget_space(dataset_id);
      // Get the dimensionality of the space
      // (unintuitively called rank in hdf5, call it space_dims here)
      space_dims = H5Sget_simple_extent_ndims(dataset_space);
      if(space_dims>1){
        planck_fail("Multidimensional filtered read not yet supported\n");
      }

      // Get size of each dimension
      std::vector<hsize_t> dim_extents(space_dims);
      std::vector<hsize_t> dim_extents_max(space_dims);
      std::vector<hsize_t> local_grid(space_dims);
      std::vector<hsize_t> start(space_dims);
      H5Sget_simple_extent_dims(dataset_space, &dim_extents[0], &dim_extents_max[0]);

      // For parallel read of grid we can only read if extent in X axis is larger than 
      if(dim_extents[0]<mpiMgr.num_ranks()){
        planck_fail("HDF5 Reader: cannot parallel read, more processors than points \
                    (or for grid data, than extent of X dim)");
      } 

      // Work out parallel read extents for each processor, 
      // either in 1d (where XYZ is provided) or 3D (where XYZ is inferred from grid)
      long np_local, np_total;
      np_total = (long)dim_extents[0];
      long step = np_total/mpiMgr.num_ranks();
      local_grid[0] = step;
      np_local=local_grid[0];
      local_grid[0] = idxs.size();
      // if(grid){
      //   if(mpiMgr.rank() == mpiMgr.num_ranks()-1) 
      //     local_grid[0] = np_total-(local_grid[0]*(mpiMgr.num_ranks()-1));

      //   // if(space_dims == 3)
      //   // {
      //   //   local_grid[1] = (int)dim_extents[1];
      //   //   local_grid[2] = (int)dim_extents[2];
      //   //   np_local = local_grid[0]*local_grid[1]*local_grid[2];
      //   //   np_total = dim_extents[0]*dim_extents[1]*dim_extents[2]; 
      //   // }    
      //   // else{
      //   //   planck_fail("HDF5 Reader: grid data expected but field is not 3 dimensional");
      //   // }
      // }
      // Starting offsets 
      //
      start[0]  = (hsize_t)(step * mpiMgr.rank());
      int new_start = step * mpiMgr.rank();
      if(redo_indexes && new_start>0){
        for(unsigned i = 0; i < idxs.size(); i++)
          idxs[i] += new_start;
        redo_indexes=false;
      }
      // if(space_dims == 3)
      // {
      //   start[1]  = 0;
      //   start[2]  = 0; 
      // }

      if (mpiMgr.master())
      {
        std::cout << "For dataset: " << d.name << std::endl;
        std::cout << "Reading " << idxs.size() << " elements of total " << np_local << std::endl;
      }
      // Read each field assuming float at the moment
      storage.resize(local_subset_length);
      //H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, &start[0], NULL, &local_grid[0], NULL); 
      H5Sselect_elements(dataset_space, H5S_SELECT_SET, local_subset_length, (const hsize_t*)&idxs[0]);
      memory_space = H5Screate_simple(space_dims, &local_grid[0], &local_grid[0]); 
      H5Dread(dataset_id, H5T_NATIVE_FLOAT, memory_space, dataset_space, H5P_DEFAULT, storage.data());
      H5Sclose(memory_space);
      H5Sclose(dataset_space); 
      H5Dclose(dataset_id);  

      // Write 

      hid_t out_dataset_space;
      hid_t subset_memory_space;
      hid_t out_dataset_id;
      //hid_t out_space_dims;


      // Open the dataset and get space
      out_dataset_id = H5Dopen1(outfile_id,d.name.c_str());
      out_dataset_space = H5Dget_space(out_dataset_id);

      // Create a memory space for the subset
      subset_memory_space = H5Screate_simple(1, &local_subset_length, NULL);
      // Select hyperslab of the dataset space
      status = H5Sselect_hyperslab(out_dataset_space, H5S_SELECT_SET, &local_subset_start, NULL, &local_subset_length, NULL); 

      // Write the storage data
      status = H5Dwrite(out_dataset_id, H5T_NATIVE_FLOAT, subset_memory_space, out_dataset_space, H5P_DEFAULT, storage.data());

      // // Work out parallel read extents for each processor, 
      // // either in 1d (where XYZ is provided) or 3D (where XYZ is inferred from grid)
      // long np_local, np_total;
      // np_total = (long)dim_extents[0];
      // long step = np_total/mpiMgr.num_ranks();
      // local_grid[0] = step;
      // np_local=local_grid[0];
      // local_grid[0] = idxs.size();
      // // if(grid){

      H5Sclose(subset_memory_space);
      H5Sclose(out_dataset_space); 
      H5Dclose(out_dataset_id); 
    }

    H5Fclose(outfile_id);
    printf("Finished writing filtered hdf5 file %s\n", filename.c_str());
    info_read = true;
    if(!was_open){
      close(params);
    } 

  }  

  void reset(paramfile& params) override {
    if(file_open){
      close(params);
    }

    std::vector<std::string>().swap(fields);
    std::vector<group>().swap(groups);
    
    // Reset state
    info_read = false;
    name = "";
  }

  void particle_fields(std::vector<std::string>& names) override {
    names = fields;
  }

  void available_fields(std::vector<std::string>& names) override {
    names.resize(0);
    for(auto g : groups)
      for(auto d : g.datasets)
        names.push_back(d.name);
  }

private:
  struct dataset{
    dataset(std::string s) : name(s) {} 
    std::string name;
    int         ndim;
  }; 
  struct group{
    group(std::string s) : name(s) {} 
    std::string           name;
    std::vector<dataset>  datasets;
  };
    
  hid_t                     file_id;
  std::vector<group>        groups;
  std::vector<std::string>  fields;
};

#endif
    // // If the data is compound
    // bool compound = params.find<bool>("hdf5_compound", false);
    // if(compound){
    //   // Create compound type for splotch particles
    //   hid_t splotch_tid;
    //   splotch_tid = H5Tcreate(H5T_COMPOUND, sizeof(particle_sim));

    //   // If XYZ are provided as fields, add them to the compound type
    //   if(fields[F_X]!="-1") {
    //     H5Tinsert(splotch_tid, fields[F_X].c_str(), HOFFSET(particle_sim, x), H5T_NATIVE_FLOAT);
    //     H5Tinsert(splotch_tid, fields[F_Y].c_str(), HOFFSET(particle_sim, y), H5T_NATIVE_FLOAT);
    //     H5Tinsert(splotch_tid, fields[F_Z].c_str(), HOFFSET(particle_sim, z), H5T_NATIVE_FLOAT);
    //   }

    //   // Insert colouring field
    //   H5Tinsert(splotch_tid, fields[F_CR].c_str(), HOFFSET(particle_sim, e.r), H5T_NATIVE_FLOAT);
    //   if(fields[F_CG]!="-1"){
    //     // colour_is_vector
    //     H5Tinsert(splotch_tid, fields[F_CG].c_str(), HOFFSET(particle_sim, e.g), H5T_NATIVE_FLOAT);
    //     H5Tinsert(splotch_tid, fields[F_CB].c_str(), HOFFSET(particle_sim, e.b), H5T_NATIVE_FLOAT);      
    //   }
    //   // If provided, insert radius and intensity fields
    //   if(fields[F_R]!="-1")
    //     H5Tinsert(splotch_tid, fields[F_R].c_str(), HOFFSET(particle_sim, r), H5T_NATIVE_FLOAT);

    //   if(fields[F_I]!="-1")
    //     H5Tinsert(splotch_tid, fields[F_I].c_str(), HOFFSET(particle_sim, I), H5T_NATIVE_FLOAT);

    //   // Get number of elements via C1 field as it is the only required field
    //   dataset_id = H5Dopen1(file_id,fields[F_CR].c_str());
    //   dataset_space = H5Dget_space(dataset_id);
    //   hsize_t np = H5Sget_simple_extent_npoints(dataset_space);
    //   H5Sclose(dataset_space); 
    //   H5Dclose(dataset_id); 

    //   // Resize splotch particles
    //   points.resize(np);
    //    printf("Splotch points: %lu", points.size());

    //   // If dataset is compound, do one read 
    //   // ...
    //   // else do n read
    //    const hsize_t start = 0;
    //    int rank = 1;
    //   if(fields[i] != "-1"){
    //     dataset_id = H5Dopen1(file_id,fields[i].c_str());
    //     dataset_space = H5Dget_space(dataset_id);
    //     H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, &start, NULL, &np, NULL); 
    //     memory_space = H5Screate_simple(rank, &np, &np); 

    //     H5Dread(dataset_id, splotch_tid, memory_space, dataset_space, H5P_DEFAULT, &points[0]);

    //     H5Sclose(memory_space);
    //     H5Sclose(dataset_space); 
    //     H5Dclose(dataset_id);       
    //   }  
    // }

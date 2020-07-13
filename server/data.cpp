#include "data.h"
using namespace OpId;
void Data::unload() {
  if(cached){
    std::vector<particle_sim>().swap(pcache);
    cached = false;
  }
  delete file;
  file = NULL;
  for(unsigned i = 0; i < n_fields; i++){
    particle_fields[i].name = "";
    particle_fields[i].normalizer = Normalizer<float>();
  }
  std::vector<unsigned>().swap(filter_id);
  std::vector<field_cache>().swap(cached_fields);
  std::vector<std::string>().swap(available_fields);
  std::vector<filter>().swap(filters);

}

void Data::read(paramfile &params, std::vector<particle_sim> &points, bool cache) {
  // If cached, return cached data
  if(cached) {
    if(filters.size()){
      // Check if any filters are marking
      // If not then we dont need to do a copy_if 
      bool do_copy_if = false;
      for(auto& f : filters){
        if(is_marking_filter(f.type))
          do_copy_if = true;
      }

      if(do_copy_if){
        points.resize(pcache.size());
        unsigned j = 0;
        for(unsigned i = 0; i < pcache.size(); i++)
        {
          if(!filter_id[i]) points[j++] = pcache[i]; 
        }
        points.resize(j);
        npoints_filtered = j;
      } else {
        npoints_filtered = pcache.size();
        points = pcache;
      }
      apply_dynamic_filters(points);
    }
    else{
      npoints_filtered = pcache.size();
      points = pcache;
    }
    return;
  }

  // Read and get field and range info
//  try{
    // Otherwise read
    if(!file) {
      create_file(params);
    }
    file->read(params, points);
  // } catch(const PlanckError& e){
  //   throw e;
  // }

  // Store particle fields
  std::vector<std::string> names(n_fields);
  file->particle_fields(names);
  for(unsigned i = 0; i < n_fields; i++){
    particle_fields[i].id   = static_cast<FieldId>(i);
    particle_fields[i].name = names[i];
  }

  // Store available fields
  file->available_fields(available_fields);

  // Check ranges
  range(points);

  if(cache){
    pcache = points;
    cached = true;
  }
}

void Data::reload_field(paramfile& params, std::string fid, std::string field_name)
{
  FieldId id = str2FieldId(fid);
  if(field_name == "" || field_name == particle_fields[id].name){
    return;
  }

  // Check if its cached
  field_cache* field = NULL;
  // First check field cache if its already there, if so grab pointer
  for(unsigned i = 0; i < cached_fields.size(); i++){
    if(cached_fields[i].first.name == field_name){
      field = &cached_fields[i];
      break;
    }
  }
  // Otherwise read field to provided vector
  if(!field){
    std::vector<float> new_field(pcache.size());
    file->read_field(params, field_name, new_field);
    field_info f;
    f.id = FieldId::NONE;
    f.name = field_name;
    range(f, new_field);
    cached_fields.push_back(std::make_pair(f, std::move(new_field)));
    field = &cached_fields.back();
  }

  // Copy to cached particle data
  float* d = field->second.data();
  switch(id)
  {
    case FieldId::F_X:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].x = d[i];
    break;
    case FieldId::F_Y:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].y = d[i];
    break;
    case FieldId::F_Z:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].z = d[i];
    break;
    case FieldId::F_CR:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].e.r = d[i];
    break;
    case FieldId::F_CG:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].e.g = d[i];
    break;
    case FieldId::F_CB:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].e.b = d[i];
    break;
    case FieldId::F_R:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].r = d[i];
    break;
    case FieldId::F_I:
      #pragma omp parallel for
      for(unsigned i = 0; i < pcache.size(); i++) pcache[i].I = d[i];
    break;
    default:
    break;
  }

  // Get the name, id and normalizer from the vector
  particle_fields[id].name = field->first.name;
  particle_fields[id].normalizer = field->first.normalizer;

}


void Data::refresh(paramfile &params, std::vector<particle_sim> &points, bool cache) {
  read(params, points, cache);
}

void Data::create_file(paramfile &params){ 
  SimTypeId sid = static_cast<SimTypeId>(params.find<int>("simtype"));
  switch(sid)
  {
    case SimTypeId::BIN_TABLE:
      planck_fail("Not implemented");
    break;  
    case SimTypeId::BIN_BLOCK:
      planck_fail("Not implemented");
    break;
    case SimTypeId::GADGET:
      planck_fail("Not implemented");
    break;
    case SimTypeId::ENZO:
      planck_fail("Not implemented");
    break;
    case SimTypeId::GADGET_MILLENIUM:
      planck_fail("Not implemented");
    break;
    case SimTypeId::BIN_BLOCK_MPI:
      planck_fail("Not implemented");
    break;
    case SimTypeId::MESH:
      planck_fail("Not implemented");
    break;
    case SimTypeId::REGULAR_HDF5:
      file = new HDF5File();
    break;
    case SimTypeId::GADGET_HDF5:
      planck_fail("Not implemented");
    break;
    case SimTypeId::VISIVO:
      planck_fail("Not implemented");
    break;
    case SimTypeId::TIPSY:
      planck_fail("Not implemented");
    break;
    case SimTypeId::H5PART:
      planck_fail("Not implemented");
    break;
    case SimTypeId::RAMSES:
      planck_fail("Not implemented");
    break;
    case SimTypeId::BONSAI:
      planck_fail("Not implemented");
    break;
    case SimTypeId::ASCII:
      planck_fail("Not implemented");
    break;
    case SimTypeId::FITS:
      planck_fail("Not implemented");
    break;
    default:
      planck_fail("Unrecognised simtype");
    break;
  }
}

void Data::range(std::vector<particle_sim> &p, FieldId fid)
{
  // Reset if necessary 
  if(fid == FieldId::F_X || fid == FieldId::NONE)
    particle_fields[FieldId::F_X].normalizer.reset();
  if(fid == FieldId::F_Y || fid == FieldId::NONE)
    particle_fields[FieldId::F_Y].normalizer.reset();
  if(fid == FieldId::F_Z || fid == FieldId::NONE)
    particle_fields[FieldId::F_Z].normalizer.reset();
  // Colours
  if(fid == FieldId::F_CR || fid == FieldId::NONE)
  particle_fields[FieldId::F_CR].normalizer.reset();
  if(particle_fields[FieldId::F_CG].name != "-1"){
    if(fid == FieldId::F_CG || fid == FieldId::NONE)
    particle_fields[FieldId::F_CG].normalizer.reset();
  if(fid == FieldId::F_CB || fid == FieldId::NONE)
    particle_fields[FieldId::F_CB].normalizer.reset();
  }
  // Intensity
  if(fid == FieldId::F_R || fid == FieldId::NONE)
    particle_fields[FieldId::F_R].normalizer.reset();
  // Radius
  if(fid == FieldId::F_I || fid == FieldId::NONE)
    particle_fields[FieldId::F_I].normalizer.reset();

  // Record ranges
  #pragma omp parallel
  {
    std::vector< Normalizer<float> > local(n_fields);
    // Parallel range
    #pragma omp for
    for (int i = 0; i < p.size(); ++i)
    {
      // Box size always recorded
      if(fid == FieldId::F_X || fid == FieldId::NONE)
        local[FieldId::F_X].collect(p[i].x);
      if(fid == FieldId::F_Y || fid == FieldId::NONE)
        local[FieldId::F_Y].collect(p[i].y);
      if(fid == FieldId::F_Z || fid == FieldId::NONE)
        local[FieldId::F_Z].collect(p[i].z);

      // Colours
      if(fid == FieldId::F_CR || fid == FieldId::NONE)
        local[FieldId::F_CR].collect(p[i].e.r);
      if(particle_fields[FieldId::F_CG].name != "-1"){
        if(fid == FieldId::F_CG || fid == FieldId::NONE)
          local[FieldId::F_CG].collect(p[i].e.g);
        if(fid == FieldId::F_CB || fid == FieldId::NONE)
          local[FieldId::F_CB].collect(p[i].e.b);
      }
      // Intensity
      if(fid == FieldId::F_R || fid == FieldId::NONE)
        local[FieldId::F_R].collect(p[i].r);
      // Radius
      if(fid == FieldId::F_I || fid == FieldId::NONE)
        local[FieldId::F_I].collect(p[i].I);
    }
    // Reduce
    #pragma omp critical
    {
      // Box size is always recorded
      if(fid == FieldId::F_X || fid == FieldId::NONE)
        particle_fields[FieldId::F_X].normalizer.collect(local[FieldId::F_X]);
      if(fid == FieldId::F_Y || fid == FieldId::NONE)
        particle_fields[FieldId::F_Y].normalizer.collect(local[FieldId::F_Y]);
      if(fid == FieldId::F_Z || fid == FieldId::NONE)
        particle_fields[FieldId::F_Z].normalizer.collect(local[FieldId::F_Z]);
      // Colours
      if(fid == FieldId::F_CR || fid == FieldId::NONE)
      particle_fields[FieldId::F_CR].normalizer.collect(local[FieldId::F_CR]);
      if(particle_fields[FieldId::F_CG].name != "-1"){
        if(fid == FieldId::F_CG || fid == FieldId::NONE)
        particle_fields[FieldId::F_CG].normalizer.collect(local[FieldId::F_CG]);
      if(fid == FieldId::F_CB || fid == FieldId::NONE)
        particle_fields[FieldId::F_CB].normalizer.collect(local[FieldId::F_CB]);
      }
      // Intensity
      if(fid == FieldId::F_R || fid == FieldId::NONE)
        particle_fields[FieldId::F_R].normalizer.collect(local[FieldId::F_R]);
      // Radius
      if(fid == FieldId::F_I || fid == FieldId::NONE)
        particle_fields[FieldId::F_I].normalizer.collect(local[FieldId::F_I]);
    }
  }
} 

void Data::range(field_info& f, std::vector<float>& v )
{
   // Record ranges
  f.normalizer.reset();
  #pragma omp parallel
  {
    Normalizer<float> local;
    // Parallel range
    #pragma omp for
    for (int i = 0; i < v.size(); ++i)
    {
      // Box size always recorded
      local.collect(v[i]);
    }
    // Reduce
    #pragma omp critical
    {
      // Box size is always recorded
      f.normalizer.collect(local);
    }
  }
}


void Data::init_filters(){
  filters.push_back(make_filter_type(&particle_fields[FieldId::F_CR], NULL, 0, particle_fields[FieldId::F_CR].name, FilterTypeId::NORM));
}

unsigned long Data::filtered_size() 
{
  return npoints_filtered;
}

// Does this filter type need a unique marker
bool Data::is_marking_filter(FilterTypeId ftid){
  switch(ftid){
    case FilterTypeId::CLIP:
      return true;
    break;
    default:
      return false;
    break;
  }
}

filter Data::make_filter_type(field_info* p_field, field_cache* d_field, int id, std::string fname, FilterTypeId ftid){
  // Assign a unique filter marker if it is a marking filter
  unsigned current_marker = filter_marker;
  if(is_marking_filter(ftid)) {
    filter_marker *= 2;
  } else {
    current_marker = 0;
  }

  // Get defaults for each filter type
  std::vector<std::string> default_args;
  switch(ftid){
    case FilterTypeId::CLIP:
    {
      field_info* fi = p_field;
      if(!fi) fi = &d_field->first;
      default_args.push_back(dataToString(fi->normalizer.minv));
      default_args.push_back(dataToString(fi->normalizer.maxv));
    }
    break;
    case FilterTypeId::COMB_FIELD:
    {
      default_args.push_back("");
      default_args.push_back(fname);
    }
    default:
    break;
  }

 std::vector<std::string> current_args = default_args;

 return filter(id, fname, ftid, default_args, current_args, current_marker);
}

void Data::create_filter(paramfile &params, std::string fieldname, FilterTypeId ftid){

  planck_assert(cached, "Cannot apply filter with no cached particles"); 
  filter_id.resize(pcache.size());

  // Assign ID
  int id = filter_identifier++;

  // Get data to filter on
  // Check if field is being used as particle field
  // If not, check if its in the cache
  // If not, load from disc
  field_info* active_particle_field = NULL;
  for(unsigned i = 0; i < n_fields; i++){
    if(particle_fields[i].name == fieldname) {
      active_particle_field = &particle_fields[i];
      break;
    }
  }
  // Put a copy of the field data into field cache
  field_cache* field = NULL;
  if(!active_particle_field){
    // First check field cache if its already there, if so grab pointer
    for(unsigned i = 0; i < cached_fields.size(); i++){
      if(cached_fields[i].first.name == fieldname){
        field = &cached_fields[i];
        break;
      }
    }
    // Otherwise read field to provided vector
    if(!field){
      std::vector<float> new_field(pcache.size());
      file->read_field(params, fieldname, new_field);
      field_info f;
      f.id = FieldId::NONE;
      f.name = fieldname;
      range(f, new_field);
      cached_fields.push_back(std::make_pair(f, std::move(new_field)));
      field = &cached_fields.back();
    }
  }

  // Swap with Normalization filter, which is always last one in the list
  // If there isnt any in the list, then this create call is for the normalization filter
  int loc = filters.size()-1;
  filters[loc] = make_filter_type(active_particle_field, field, id, fieldname, ftid);
  filters.push_back(make_filter_type(&particle_fields[FieldId::F_CR], NULL, -1, particle_fields[FieldId::F_CR].name, FilterTypeId::NORM));
}


// // Add filter on data field
// nvoid Data::add_dynamic_filter(paramfile &params, std::string fieldname, FilterTypeId ftid, std::vector<float> args) {
//   planck_assert(cached, "Cannot apply filter with no cached particles"); 
//   filter_id.resize(pcache.size());

//   // Check filter size throw error if larger than sizeof(unsigned)*8 

//   // Check if field is being used as particle field
//   field_info* active_particle_field = NULL;

//   for(unsigned i = 0; i < n_fields; i++){
//     if(particle_fields[i].name == fieldname) {
//       active_particle_field = &particle_fields[i];
//       break;
//     }
//   }
//   // If not, check the cache
//   // If not, load from disc

//   // Cache a copy of the field data into field cache
//   field_cache* field = NULL;
//   if(!active_particle_field){
//     // First check field cache if its already there, if so grab pointer
//     for(unsigned i = 0; i < cached_fields.size(); i++){
//       if(cached_fields[i].first.name == fieldname){
//         field = &cached_fields[i];
//         break;
//       }
//     }
//     // Otherwise read field to provided vector
//     if(!field){
//       std::vector<float> new_field(pcache.size());
//       file->read_field(params, fieldname, new_field);
//       field_info f;
//       f.id = FieldId::NONE;
//       f.name = fieldname;
//       range(f, new_field);
//       cached_fields.push_back(std::make_pair(f, std::move(new_field)));
//       field = &cached_fields.back();
//     }
//   }
  
//   // If its a static filter, we apply straight away
//   if(is_static_filter(ftid)){
//    if(active_particle_field){
//       // Apply static filter to particle field
//         std::vector<float> defaults;
//         switch(ftid)
//         {
//             case FilterTypeId::CLIP:
//             {
//               defaults.push_back(active_particle_field->normalizer.minv);
//               defaults.push_back(active_particle_field->normalizer.maxv);
//               float min, max;
//               // If args are passed, use them, otherwise use normalizer defaults
//               if(args.size() != 2){ 
//                 args = defaults;
//                 break;
//               }
//               min = args[0];
//               max = args[1];
//               auto f = [min, max](float element) {
//                 return (element < min || element > max);
//               };

//               // Apply filter
//               filter_if(active_particle_field.id, f, filter_marker);
//             }
//             break;
//             default:
//             break;
//         }

//     } else {
//       // Apply static filter to data field
//         std::vector<float> defaults;
//         switch(ftid)
//         {
//             case FilterTypeId::CLIP:
//             {
//               defaults.push_back(field->first.normalizer.minv);
//               defaults.push_back(field->first.normalizer.maxv);
//               float min, max;
//               // If args are passed, use them, otherwise use normalizer defaults
//               // If we use normalizer defaults we dont actually need to apply the filter...
//               if(args.size() != 2){ 
//                 args = defaults;
//                 break;
//               }
//               min = args[0];
//               max = args[1];
//               auto f = [min, max](float element) {
//                 return (element < min || element > max);
//               };
//               filter_if(field->second.data(), f, filter_marker);
//             }
//             break;
//             default:
//             break;
//         }
//     }
//   }

//   // For STATIC filters we apply straight away
//   // Make a functor for the filterID with a pointer to array of fieldname
//   // pass it to filter_if


//   int id = filters.size();
//   filters.push_back(filter(id, fieldname, ftid, defaults, args, filter_marker));

//   // Increment the filter marker if this type of filter uses it
//   if(is_marking_filter(ftid))
//     filter_marker *= 2;
// }

void Data::update_dynamic_filter(paramfile &params, const std::vector<std::string>& args){
  // Get id from args
  int id;
  stringToData(args[0], id);

  // Get filter from filter list based on id
  // Predicate to match filter id
  auto pred = [id](const filter& f) -> bool{ 
    return (f.id == id);
  };

  // Find the matching filter
  auto found = find_if(filters.begin(), filters.end(), pred);
  if(found == filters.end()) {
    printf("Warning - updating non-existent filter\n");
    return;
  }
  auto& flt = *found;


  // Check if we have the data in particle cache already
  field_info* particle_field = NULL;
  for(unsigned i = 0; i < n_fields; i++){
    if(particle_fields[i].name == flt.name) {
      particle_field = &particle_fields[i];
      break;
    }
  }

  // If not, check the data cache
  field_cache* data_field = NULL;
  if(!particle_field){
    // First check field cache if its already there, if so grab pointer
    for(unsigned i = 0; i < cached_fields.size(); i++){
      if(cached_fields[i].first.name == flt.name){
        data_field = &cached_fields[i];
        break;
      }
    }
    // Otherwise read field to provided vector
    if(!data_field){
      std::vector<float> new_field(pcache.size());
      file->read_field(params, flt.name, new_field);
      field_info f;
      f.id = FieldId::NONE;
      f.name = flt.name;
      range(f, new_field);
      cached_fields.push_back(std::make_pair(f, std::move(new_field)));
      data_field = &cached_fields.back();
    }
  }

  ///// break here and see waht happens...
  if(particle_field){
    // Apply filter to particle field
    apply_particle_filter(params, particle_field, flt, args);
  } else {
    // Apply filter to data field
    apply_data_filter(params, data_field, flt, args);
  }
}

void Data::remove_dynamic_filter(int id)
{
  // Predicate to match filter id
  auto pred = [id](const filter& f) -> bool{ 
    return (f.id == id);
  };

  // Find the matching filter, unfilter data and remove it from the list
  auto flt = find_if(filters.begin(), filters.end(), pred);
  unfilter(flt->marker);
  remove_if(filters.begin(), filters.end(), pred);
}

void Data::unfilter(unsigned marker){
  #pragma omp parallel for
  for(unsigned i = 0; i < pcache.size(); i++)
    filter_id[i] &= ~marker;
}

void Data::apply_particle_filter(paramfile& params, field_info* particle_field, filter& flt, const std::vector<std::string>& args){
  // Switch on filter type

  switch(flt.type)
  {
      case FilterTypeId::CLIP:
      {
        planck_assert(args.size()==3, "CLIP filter must have args: id, min, max");
        float min, max;
        stringToData(args[1], min);
        stringToData(args[2], max);
        auto& filter_id_ref = filter_id;
        auto f = [&filter_id_ref, &flt, min, max](float element, unsigned i) {
          filter_id_ref[i] &= ~flt.marker;
          if(element < min || element > max)
            filter_id_ref[i] |= flt.marker;
        };
        apply_filter(pcache, particle_field->id, f);
        flt.args[0] = dataToString(min);
        flt.args[1] = dataToString(max);
      }
      break;
      case FilterTypeId::COMB_FIELD:
      {
        planck_assert(args.size()==3, "COMB filter must have args: id, operation, field");

        // Load field
        field_cache* data_field = NULL;
        std::string rhs_field = args[2];
        // First check field cache if its already there, if so grab pointer
        for(unsigned i = 0; i < cached_fields.size(); i++){
          if(cached_fields[i].first.name == rhs_field){
            data_field = &cached_fields[i];
            break;
          }
        }
        // Otherwise read field to provided vector
        if(!data_field){
          std::vector<float> new_field(pcache.size());
          file->read_field(params, rhs_field, new_field);
          field_info f;
          f.id = FieldId::NONE;
          f.name = rhs_field;
          range(f, new_field);
          cached_fields.push_back(std::make_pair(f, std::move(new_field)));
          data_field = &cached_fields.back();

        }  

        // Load operation
        OperationId oid = Str2OpId(args[1]);
        auto op = operations[static_cast<int>(oid)];

        // Build function
        auto field_ref = data_field->second.data();
        auto f = [&field_ref, op](float& element, unsigned i) {
          element = op(element, field_ref[i]);
        };

        apply_filter(pcache, particle_field->id, f);
        flt.args[0] = args[1];
        flt.args[1] = args[2];

        // Re range
        range(pcache, particle_field->id);
      }
      break;     
      default:
      break;    
  }
}

void Data::apply_data_filter(paramfile& params, field_cache* data_field, filter& flt, const std::vector<std::string>& args){
  // Switch on filter type
  switch(flt.type)
  {
      case FilterTypeId::CLIP:
      {
        planck_assert(args.size()==3, "CLIP filter must have args: id, min, max");
        float min, max;
        stringToData(args[1], min);
        stringToData(args[2], max);
        auto f = [min, max](float element) {
          return (element < min || element > max);
        };
        filter_if(data_field->second.data(), f, flt.marker);
        flt.args[0] = dataToString(min);
        flt.args[1] = dataToString(max);
      }
      break;
      case FilterTypeId::COMB_FIELD:
      {

        planck_assert(args.size()==3, "COMB filter must have args: id, operation, field");

        // Load field
        field_cache* data_field = NULL;
        std::string rhs_field = args[2];
        // First check field cache if its already there, if so grab pointer
        for(unsigned i = 0; i < cached_fields.size(); i++){
          if(cached_fields[i].first.name == rhs_field){
            data_field = &cached_fields[i];
            break;
          }
        }
        // Otherwise read field to provided vector
        if(!data_field){
          std::vector<float> new_field(pcache.size());
          file->read_field(params, rhs_field, new_field);
          field_info f;
          f.id = FieldId::NONE;
          f.name = rhs_field;
          range(f, new_field);
          cached_fields.push_back(std::make_pair(f, std::move(new_field)));
          data_field = &cached_fields.back();

        }  

        // Load operation
        OperationId oid = Str2OpId(args[1]);
        auto op = operations[static_cast<int>(oid)];

        // Build function
        auto field_ref = data_field->second.data();
        auto f = [&field_ref, op](float& element, unsigned i) {
          element = op(element, field_ref[i]);
        };

        apply_filter(data_field->second.data(), f);
        flt.args[0] = args[1];
        flt.args[1] = args[2];

        // Re range
        range(data_field->first, data_field->second);
      }
      break;     
      default:
      break;    
  }
}

void Data::apply_dynamic_filters(std::vector<particle_sim> &points){
  // For each dynamic filter
  // Determine if it applies to a used quantity
  for(auto filter : filters){
    FieldId fid;
    int found = -1;
    for(auto field : particle_fields)
    {
       if(filter.name == field.name){
          fid = field.id;
          found = static_cast<int>(fid);
       }
    } 
    if(found>=0){
       // Make a functor for the filterID with a pointer to array of fieldname
      // pass to apply_filter
      switch(filter.type)
      {
          case FilterTypeId::NORM:
          {
            float min = particle_fields[found].normalizer.minv;
            float max = particle_fields[found].normalizer.maxv;
            auto f = [min, max](float& element, unsigned i) {
              element = (element-min)/(max-min);
            };
            apply_filter(points, fid, f);
          }
          break;
          // case FilterTypeId::LOG:
          // {
          //   // auto f = [](float& element) { 
          //   //   if(element < 0)
          //   //     element = -37;
          //   //   else
          //   //     element = log10(element);
          //   // };
          //   // apply_filter(points, fid, f);
          //   // float min = particle_fields[found].normalizer.minv;
          //   // float max = particle_fields[found].normalizer.maxv;
          //   // particle_fields[found].normalizer.minv = ;
          //   // particle_fields[found].normalizer.minv = 1;
          // }
          // break;
          default:
          break;
      }         
    } else {
      // Apply filter based on cached field

    }
  }
}

bool Data::write_filtered(paramfile& params, std::string filename){

  if(filter_io_future.valid() && filter_io_future.wait_for(std::chrono::microseconds(1)) != std::future_status::ready){
    return false;
  } else {
    bool previous_result;
    if(filter_io_future.valid())
      previous_result = filter_io_future.get();
    filter_io_future = std::async( std::launch::async, [this, params, filename]() mutable -> bool { 
                // Dont let anyone do filtering IO when were already doing it
                std::lock_guard<std::mutex> lock(filter_io_mutex);
                std::vector<unsigned long long> idxs;
                {
                  // Dont let users modify filters while were preparing for filter IO
                  std::lock_guard<std::mutex> lock(filter_prep_mutex);
                  // Create indexes from filtering list
                  idxs.resize(filtered_size());
                  for(unsigned i = 0, j = 0; i < filter_id.size(); i++){
                    if(filter_id[i]==0)
                      idxs[j++] = i; 
                  }
                }
                file->write_filtered(params, filename, idxs);
                return true;
              }); 
  }

  return true;
}









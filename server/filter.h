#ifndef SPLOTCH_FILTER_H
#define SPLOTCH_FILTER_H


namespace OpId {
enum class OperationId: int {NOP = 0, ADD = 1, SUBTRACT = 2, MULTIPLY = 3, DIVIDE = 4};

static std::vector<std::string> ops = {
  "",
  "+",
  "-",
  "*",
  "/",
};

static std::map<OperationId, std::string> o2s = {
    {OperationId::NOP,          ""},
    {OperationId::ADD,          "+"},
    {OperationId::SUBTRACT,     "-"},
    {OperationId::MULTIPLY,     "*"},
    {OperationId::DIVIDE,       "/"},
};

static std::string OpIdToStr(OperationId id) {
    if(id < OperationId::NOP || id > OperationId::DIVIDE){
       printf("Server: OperationId() invalid op id %d\n", id);
       return o2s[OperationId::NOP];
     }
    return o2s[id];
}

static OperationId Str2OpId(std::string str) {
    auto it = o2s.begin();
    while(it != o2s.end() && it->second != str) it++;
    planck_assert(it!=o2s.end(), "Str2OpId() string doesnt exist");
    return it->first;
}
}


enum class FilterTypeId: int {NONE = -1, CLIP = 0, COMB_FIELD = 1, NORM = 2};
static std::map< FilterTypeId, std::string > ftid2s = {
          {FilterTypeId::NONE,        "NONE"},
          {FilterTypeId::CLIP,        "CLIP"},
          {FilterTypeId::COMB_FIELD,  "COMB_FIELD"},
          {FilterTypeId::NORM,        "NORM"}
};
static std::string FilterTypeId2str(FilterTypeId fid) {
    auto it = ftid2s.find(fid);
    planck_assert(it!=ftid2s.end(), "FilterTypeId2str() field id doesnt exist");
    return it->second;
}
static FilterTypeId Str2FilterTypeId(std::string str) {
    auto it = ftid2s.begin();
    while(it != ftid2s.end() && it->second != str) it++;
    planck_assert(it!=ftid2s.end(), "str2FilterTypeId() string doesnt exist");
    return it->first;
}
struct filter
{
  filter(int _id, std::string _name, FilterTypeId _ftid, const std::vector<std::string>& _defs, const std::vector<std::string>& _args, unsigned _marker)
  {
    id   = _id;
    name = _name;
    type = _ftid;
    defaults = _defs;
    args = _args;
    marker = _marker;
  }

  std::string name;
  FilterTypeId type;
  int id;
  std::vector<std::string> defaults;
  std::vector<std::string> args;
  unsigned marker;
};

#endif
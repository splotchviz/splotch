#ifndef OBSERVE_H
#define OBSERVE_H

#include <functional>
#include <vector>

//
// observe.h implements an observer/observable pattern via templates 
// rather than inheritence, allowing the user to bind any type of callback 
// to an observer object which is held for the lifetime of the observer.
// Observers and observables are aware of each other, allowing automatic 
// detachment if either goes out of scope.
//

// Forward declare Observer
template<typename ...NotifyArgs>
class Observable;

// Observer class
// Attaches to an observable and binds a user provided callback void notify(...)
// Templated on the arguments to notify()
template <typename ...NotifyArgs>
class Observer
{
  friend class Observable< NotifyArgs ...>;
  using NotifyFunction = std::function<void(const NotifyArgs& ...args)>;
  using ObservableList   = std::vector< Observable<NotifyArgs...>* >;
  public:
  Observer() {}
  ~Observer() {
    detach_all();
  }

  // Bind a user callback to the observer
  // Can be lambda, function pointer, or std::function etc
  template<typename ClassType>
  void bind_callback(ClassType&& c) {
    notify = NotifyFunction(std::forward<ClassType>(c));
  }
 
  // Detach from a specific observable
  void detach(Observable< NotifyArgs... >* to_detach) {
    auto drg = [to_detach, this](Observable< NotifyArgs... >* current){ 
      if(current==to_detach) {
        to_detach->detach(this); 
        return true;
      }
      return false;
    };
    obs.erase( std::remove_if(obs.begin(), obs.end(), drg), obs.end());
  }

  // Detach from all observables
  void detach_all() {
    // For each observer in list, detach and remove from list
    auto drg = [this](Observable< NotifyArgs... >* current){ 
      current->detach(this); 
      return true;
    };
    obs.erase( std::remove_if(obs.begin(), obs.end(), drg), obs.end());
  }

  private:
  // Attach to an observable, only an observable should call this
  void attach(Observable< NotifyArgs... >* to_attach) {
    obs.push_back(to_attach);
  }

  NotifyFunction notify;
  ObservableList obs;
};

// Observable class
// Can be attached to by observer objects, and calls each attach observer 
// callback when observable::notify() is called
// Templated on the arguments to the notify() function
template <typename ...NotifyArgs>
class Observable
{
  friend class Observer< NotifyArgs ... >;
public:
  using NotifyFunction = std::function<void(const NotifyArgs& ...args)>; 
  using ObserverList   = std::vector<Observer< NotifyArgs... >*>;

  Observable() {}
  ~Observable() {
    detatch_all();
  }

  // Attach an observer
  void attach(Observer< NotifyArgs ... >& to_attach){
    to_attach.attach(this);
    obs.push_back(&to_attach);
  }

  // Detatch a specific observer
  void detach(Observer< NotifyArgs ... >& to_detach){
      to_detach.detach(this);      
  }

  // Detatch all observers
  void detatch_all() {
    auto observer = obs.rbegin();
    while(observer.base() != obs.end()) {
      (*observer)->detach(this);
      observer = obs.rbegin();
    }
  }

  // Notify all observers
  void notify(NotifyArgs... args) {
    for(auto observer: obs){
      observer->notify(args...);
    }
  }

private:
  // This is the real detach, i.e. it removes an observer from our local list
  // Only an observer should call this
  void detach(Observer< NotifyArgs ... >* to_detach) {
    obs.erase( std::remove(obs.begin(), obs.end(), to_detach), obs.end());
  }

  ObserverList obs;
};

#endif
/*
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *      Tim Dykes 
 *      University of Portsmouth
 *
 */
 
#include "RemoteViewer.h"
#include "previewer/Previewer.h"
#include "cxxsupport/ls_image.h"
#include <cstring>
namespace previewer
{

//! Specialization for JPEGImage from tjpp library
struct SerializeJPEGImage {
    using SS = srz::SerializePOD< size_t >;
    static srz::ByteArray Pack(const tjpp::JPEGImage& jpi,
                          srz::ByteArray buf = srz::ByteArray()) {
        buf = srz::PackArgs(jpi.Width(),jpi.Height(),jpi.PixelFormat(),jpi.ChrominanceSubSampling(),jpi.Quality(),jpi.CompressedSize());
        int sz = buf.size();
        buf.resize(sz+jpi.CompressedSize());
        memcpy(&buf[sz],jpi.DataPtr(),jpi.CompressedSize());
        return buf;
    }
    static srz::ConstByteIterator UnPack(srz::ConstByteIterator bi,
                                    tjpp::JPEGImage& jpi) {
      int w,h,pf,ss,q;
      size_t sz;
      bi = srz::SerializePOD<int>::UnPack(bi, w);
      bi = srz::SerializePOD<int>::UnPack(bi, h);
      bi = srz::SerializePOD<int>::UnPack(bi, pf);
      bi = srz::SerializePOD<int>::UnPack(bi, ss);
      bi = srz::SerializePOD<int>::UnPack(bi, q);
      bi = srz::SerializePOD<size_t>::UnPack(bi, sz);
      TJPF pf1 = (TJPF)pf;
      TJSAMP ss1 = (TJSAMP)ss;

      jpi.Reset(w,h,pf1,ss1,q);
      planck_assert(UncompressedSize(jpi) >= sz,"srz::UnPack: allocated buffer < recieved JPEGSize");
      memmove(jpi.DataPtr(),&(*bi),sz);
      jpi.SetCompressedSize(sz);
      return bi;
    }
};
 //! De-serialize data from byte array.
void UnPack(const srz::ByteArray& ba, tjpp::JPEGImage& d) {
    SerializeJPEGImage::UnPack(ba.begin(), d);
}

    void RemoteViewer::Load(const ParticleData& pData)
    {
        DebugPrint("Loading RemoteViewer renderer\n");

        // All we need for remote viewing is the parameter file
        splotchParams = Previewer::parameterInfo.GetParamFileReference();

        // Alloc some memory for splotch to render to
        xres = splotchParams->find<int>("xres");
        yres = splotchParams->find<int>("yres");

        // camera.SetMainCameraStatus(true);

        renderMaterial = new FF_ParticleMaterial();
        renderMaterial->Load();
        renderMaterial->SetTexture(true);
        // Load a texture (perhaps one that says 'no image recieved')
        renderMaterial->LoadTexture(ParticleSimulation::GetExePath()+"previewer/data/textures/remote_viewer_pending.tga", GL_TEXTURE_2D);
        image_recieved=false;
        first_image=true;
        running=true;

        // Get unqiue resource identifiers from parameters
        image_uri = splotchParams->find<std::string>("image_uri");
        event_uri = splotchParams->find<std::string>("event_uri");

        ac.Start(event_uri.c_str());

        int size =xres*yres*3;
        recv_thread = std::thread(&RemoteViewer::Reciever, this, size);
        event_thread = std::thread(&RemoteViewer::Sender, this);

        ft.resize(10);
        ft.start(0);
    }

    void RemoteViewer::Draw()
    {   
        // Draw image
        // Set 3d rendering viewport size */
        float viewXmin = ParticleSimulation::GetRenderXMin();
        float viewYmin = ParticleSimulation::GetRenderYMin();

        // Set 3d render viewport to scissor and clear
        glViewport(viewXmin,viewYmin, ParticleSimulation::GetRenderWidth(), ParticleSimulation::GetRenderHeight());

        // Bind material
        renderMaterial->Bind();

        glBegin(GL_QUADS);

            glTexCoord2f(0,0);  glVertex3f(-1, -1, 0);
            glTexCoord2f(0,1);  glVertex3f(-1, 1, 0);
            glTexCoord2f(1,1);  glVertex3f(1, 1, 0);
            glTexCoord2f(1,0);  glVertex3f(1, -1, 0);

        glEnd();
        
        renderMaterial->Unbind();

        if(!image_queue.Empty()) ParticleSimulation::rendererUpdated = false;
        ctr++;

    }

    void RemoteViewer::Unload()
    {
        running=false;
        // Push to event queue to allow event thread to register running==false
        event_queue.Push(Event());
        is.Stop();
        recv_thread.join();
        event_thread.join();
        ac.Stop();
        delete renderMaterial;
    }

    void RemoteViewer::Update()
    {
        int ic = image_queue.Size();
        if(ic>0)
        {
            // Some timing stuff
            if(!(ctr%100)) 
            {
                ft.mark(0, "Time since last frame: ms ");
                if(ic>2) printf("Warning: Image queue size: %u\n", ic);
            }
            else
            {
                ft.mark(0);
            }

            tjpp::Image im(std::move(image_queue.Pop()));

            if(first_image)
            {

                renderMaterial->LoadTexture(im.DataPtr(), xres, yres, GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE); 
                first_image=false;
            }
            else
            {
                renderMaterial->ReplaceTexture(im.DataPtr(), xres, yres); 
            }
            if(ic>1) ParticleSimulation::rendererUpdated = false;
            if(ctr > 1024) ctr-=1024;
        }
    }

    void RemoteViewer::OnKeyPress(Event ev)
    {
        event_queue.Push(ev);
    }

    void RemoteViewer::OnKeyRelease(Event ev)
    {
        event_queue.Push(ev);
    }

    void RemoteViewer::OnMotion(Event ev)
    {
        event_queue.Push(ev);
    } 

    void RemoteViewer::OnButtonPress(Event ev)
    {
        event_queue.Push(ev);
    } 

    void RemoteViewer::OnButtonRelease(Event ev)
    {
        event_queue.Push(ev);
    } 

    void RemoteViewer::SetRenderBrightness(unsigned type, float b) 
    {
        // previewer::Event ctlEv;
        // ctlEv.id = previewer::evServerControl;
        // ctlEv.field0 = (float)type;
        // ctlEv.field1 = b;
        // strcpy(ctlEv.desc,"sctrl_set_brightness");
        // event_queue.Push(ctlEv);
    }
    void RemoteViewer::SetSmoothingLength(unsigned type, float sl) 
    {
        // previewer::Event ctlEv;
        // ctlEv.id = previewer::evServerControl;
        // ctlEv.field0 = (float)type;
        // ctlEv.field1 = sl;
        // strcpy(ctlEv.desc,"sctrl_set_smoothing");
        // event_queue.Push(ctlEv);
    }

    void RemoteViewer::PrintCamera()
    {
        // previewer::Event ctlEv;
        // ctlEv.id = previewer::evServerControl;
        // ctlEv.field0 = 0;
        // ctlEv.field1 = 0;
        // strcpy(ctlEv.desc,"sctrl_print_camera");
        // event_queue.Push(ctlEv);        
    }

    void RemoteViewer::Reciever(int size) 
    {
        // Listen for incoming images, buffer size is max uncompressed image
        if(!is.Started()) is.Start(image_uri.c_str(), size, 15000);

        tjpp::TJDeCompressor decomp(1920 * 1080 * 4 * 4); //up to RGBX 4k support
        tjpp::JPEGImage jpi;

        // use uri = "tcp://<hostname or address>:port" to connect
        is.Loop([this,size,&decomp,&jpi](const std::vector< char >& v) {
            if(!v.empty()) 
            {
                //printf("RemoteViewer: recieved %lu bytes (v.size()), max buffer size %d\n",v.size(), size);
                //ft.mark(1, "Time since last recieve: us");
                UnPack(v, jpi);
                image_queue.Push(decomp.DeCompress(jpi.DataPtr(), jpi.CompressedSize(), jpi.PixelFormat()));

                //printf("queue.size(): %i\n",image_queue.Size());
                ParticleSimulation::rendererUpdated = false;
            }
            return !v.empty(); 
        });
    }



// //! \c std::string serialization.
// struct SerializeEvent {
//     static srz::ByteArray Pack(const Event& e,
//                           srz::ByteArray buf = srz::ByteArray()) {
//      buf = srz::SerializePOD<int>::Pack(e.eventType,buf);
//      buf = srz::SerializeString::Pack(e.keyID,buf);
//      buf = srz::SerializePOD<float>::Pack(e.mouseX,buf);
//      buf = srz::SerializePOD<float>::Pack(e.mouseY,buf);
//         return buf;
//     }
//     static srz::ConstByteIterator UnPack(srz::ConstByteIterator bi, Event& e) {
//         bi =  srz::SerializePOD<int>::UnPack(bi, e.eventType);
//      bi = srz::SerializeString::UnPack(bi, e.keyID);
//      bi = srz::SerializePOD<float>::UnPack(bi, e.mouseX);
//      bi = srz::SerializePOD<float>::UnPack(bi, e.mouseY);
//         return bi;
//     }
// };
 //! De-serialize data from byte array.
// Event UnPack(const srz::ByteArray& ba) {
//     Event d;
//     SerializeEvent::UnPack(ba.begin(), d);
//     return d;
// }
    void RemoteViewer::Sender()
    {
        while(running)
        {
            previewer::Event ev = event_queue.Pop();

            if(ev.eventType==evIgnoreEvent)
            {
                // 
            }
            else
            {
                // Convert Splotch event to server event for sending
                int eid = 0;
                srz::ByteArray data;
                switch(ev.eventType)
                {
                    case previewer::evButtonPress:
                    {   
                        eid = 1;
                        data = srz::SerializePOD<int>::Pack(eid,data);
                        // Mouse x, y 
                        data = srz::SerializePOD<int>::Pack((int)ev.mouseX,data);
                        data = srz::SerializePOD<int>::Pack((int)ev.mouseY,data);
                        // Button ID
                        std::stringstream ss(ev.keyID);
                        int bid;
                        ss >> bid;
                        // Convert to server format
                        switch(bid)
                        {
                            case 1: 
                                bid = 0;
                                break;
                            case 2: 
                                bid = 4;
                                break;
                            case 3:
                                bid = 2; 
                                break;
                        }
                        data = srz::SerializePOD<int>::Pack(bid,data);
                    }
                    break;
                    case previewer::evButtonRelease:
                    {
                        eid = 2;
                        data = srz::SerializePOD<int>::Pack(eid,data);
                        // Mouse x, y
                        data = srz::SerializePOD<int>::Pack((int)ev.mouseX,data);
                        data = srz::SerializePOD<int>::Pack((int)ev.mouseY,data);
                        // Button ID
                        std::stringstream ss(ev.keyID);
                        int bid;
                        ss >> bid;
                        // Convert to server format
                        switch(bid)
                        {
                            case 1: 
                                bid = 0;
                                break;
                            case 2: 
                                bid = 4;
                                break;
                            case 3:
                                bid = 2; 
                                break;
                        }
                        data = srz::SerializePOD<int>::Pack(bid,data);
                    }
                    break;
                    case previewer::evMouseMotion:
                    {
                        eid = 3;
                        data = srz::SerializePOD<int>::Pack(eid,data);
                        // Mouse x, y
                        data = srz::SerializePOD<int>::Pack((int)ev.mouseX,data);
                        data = srz::SerializePOD<int>::Pack((int)ev.mouseY,data);
                        // Button ID
                        std::stringstream ss(ev.keyID);
                        int bid;
                        ss >> bid;
                        // Convert to server format
                        switch(bid)
                        {
                            case 1: 
                                bid = 0;
                                break;
                            case 2: 
                                bid = 4;
                                break;
                            case 3:
                                bid = 2;
                                break; 
                        }
                        data = srz::SerializePOD<int>::Pack(bid,data);
                    }
                    break;
                    case previewer::evKeyPress:
                    {
                        eid = 4;
                        data = srz::SerializePOD<int>::Pack(eid,data);
                        int keyID = pv2ascii(ev.keyID);
                        data = srz::SerializePOD<int>::Pack(keyID,data);
                        // Ignore modifier for now
                        int modifier = 0;
                        data = srz::SerializePOD<int>::Pack(modifier,data);
                    }
                    break;   
                    case previewer::evKeyRelease:
                    {
                        eid = 5;
                        data = srz::SerializePOD<int>::Pack(eid,data);
                        int keyID = pv2ascii(ev.keyID);
                        data = srz::SerializePOD<int>::Pack(keyID,data);
                        // Ignore modifier for now
                        int modifier = 0;
                        data = srz::SerializePOD<int>::Pack(modifier,data);
                    }
                    break;   
                    case previewer::evServerControl:
                        eid = 8;
                        data = srz::SerializePOD<int>::Pack(eid,data);

                    break;
                    case previewer::evQuitApplication:
                        // For quit we send a NULL event, might want to use this at some point...
                        eid = 0;
                        data = srz::SerializePOD<int>::Pack(eid,data);
                    default:
                     printf("RemoteViewer: Warning, event not recognised.\n");
                    break;
               }

                //srz::ByteArray outBuf = SerializeEvent::Pack(ev);
                ac.SendNoReply(data);
            }
        }
    }

    int RemoteViewer::pv2ascii(std::string pvKey)
    {
        // Convert previewer keyID to ascii
        int asciiKey;
        // If its a single character, conversion is simple
        if(pvKey.size() == 1)
            asciiKey = pvKey[0];
        else
        {
            // Process the special keys
            // Should use ascii in previewer too at some point to remove the need for this
            if(pvKey == "RETURN")
                asciiKey = 13;
            else if(pvKey == "BACKSPACE")
                asciiKey = 8;
            else if( pvKey == "TAB")
                asciiKey = 9;
            else if(pvKey == "ESCAPE")
                asciiKey = 27;
            else if( pvKey == "DELETE")
                asciiKey = 127;
            else
                asciiKey = 0;
            // These are the other special keys, ignore them for now 
            // else if(keyPressed == "SHIFT")
            //     asciiKey = ;
            // else if( keyPressed == "CTRL")
            //     asciiKey = "CTRL";
            // else if( keyPressed == "ALT")
            //     asciiKey = "ALT";
            // else if(keyPressed == "LEFT")
            //     asciiKey = "LEFT";
            // else if(keyPressed == "RIGHT")
            //     asciiKey = "RIGHT";
            // else if(keyPressed == "UP")
            //     asciiKey = "UP";
            // else if(keyPressed == "DOWN")
            //     asciiKey = "DOWN";
        }
        return asciiKey;
    }

}


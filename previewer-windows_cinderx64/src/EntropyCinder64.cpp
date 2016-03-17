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
* 		Elliott Ayling
* 		University of Portsmouth
*
*/

#include <fstream>
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/CameraUi.h"
#include "cinder/params/Params.h"
#include "cinder/Log.h"
#include "previewer_n\application.h"
#include "renderers\CinderRender.h"

using namespace ci;
using namespace ci::app;
using namespace std;

const int WIDTH = 1920;
const int HEIGHT = 1080;
static string pathString;
typedef std::map<std::string, std::string> params_type;

class SplotchCinder : public App {
public:
	static void prepareSettings(Settings *settings);
	void setup() override;
	void mouseDown(MouseEvent event) override;
	void mouseDrag(MouseEvent event) override;
	void mouseMove(MouseEvent event) override;
	void update() override;
	void draw() override;
	void loadMainScene();
	void createParams(bool, bool);
	void toggleBlur();
	void setParamString(const string &key, const string &value);
	void writeToParameterFile();

	CameraUi				mCamUi;
	params::InterfaceGlRef	mParams;

	previewer::Previewer	Program;
	CinderRender			renderer;

private:
	ivec2					mLastMousePos;
	string					paramFileArg;
	std::ofstream			cinder;
	bool					drawMainScene;
	paramfile*				myParamFile;
	vector<string>			paramSettings;
	string					**paramArray;
	params_type				allParams;
};

void SplotchCinder::prepareSettings(Settings *settings)
{
	settings->setWindowSize(WIDTH, HEIGHT);
	settings->setResizable(false);

	vector<std::string> parExtension(1, "par");

	try {
		fs::path path = ci::app::getOpenFilePath("", parExtension);
		if (!path.empty()) {
			pathString = path.generic_string();
		}
	}
	catch (Exception &exc) {
		CI_LOG_EXCEPTION("failed to find parameter file.", exc);
	}
}

void SplotchCinder::setup()
{
	vector<std::string> parExtension(1, "par");
	drawMainScene = false;

	paramFileArg = pathString;
	try
	{
		if (previewer::FileLib::FileExists(const_cast<char*>(paramFileArg.c_str())))
		{
			myParamFile = Program.LoadParameterFile(paramFileArg);
			mParams = params::InterfaceGl::create(getWindow(), "Parameters", toPixels(ivec2(200, 300)));
			createParams(false, true);
		}
	}
	catch (Exception &exc)
	{
		CI_LOG_EXCEPTION("failed to load parameter file.", exc);
	}

}

void SplotchCinder::loadMainScene()
{
	try
	{
		previewer::ParticleData loadedParticles;
		loadedParticles = Program.Load();
		renderer.Load(loadedParticles);
		mParams->clear();
		delete[] paramArray;
		createParams(true, true);
		drawMainScene = true;
	}
	catch (Exception &exc)
	{
		CI_LOG_EXCEPTION("failed to load particles.", exc);
	}
}

void SplotchCinder::createParams(bool mainScene, bool paramLoad)
{
	if (mainScene)
	{
		string numberOfParticles = "Number of particles: " + to_string (renderer.numberOfParticles);
		mParams->addText(numberOfParticles);
		mParams->addParam("Particle Brightness", &renderer.brightnessMod).min(0.1f).max(10.0f).precision(2).step(0.02f);
		mParams->addParam("Particle Size Modifier", &renderer.particleSize).min(0.0f).max(3.0f).precision(2).step(0.02f);
		mParams->addParam("Saturation", &renderer.saturation).min(0.0f).max(2.0f).precision(2).step(0.01f);
		mParams->addParam("Contrast", &renderer.contrast).min(0.0f).max(2.0f).precision(2).step(0.01f);
		mParams->addButton("Blur", bind(&SplotchCinder::toggleBlur, this));
		mParams->addParam("Blur Strength", &renderer.blurStrength).min(0.0f).max(1.0f).precision(2).step(0.01f);
		mParams->addParam("Blur Color Modifier", &renderer.blurColorModifier).min(0.0f).max(2.0f).precision(2).step(0.01f);
		mParams->addSeparator();
	}
	if (paramLoad)
	{	
		allParams = myParamFile->getParams();
		string first, second;
		int count = 0;
		paramArray = new string*[allParams.size()];

		for (int i = 0; i < allParams.size(); ++i)
		{
			paramArray[i] = new string[2];
		}

		if (mainScene) mParams->addButton("Reload", bind(&SplotchCinder::loadMainScene, this));
		else mParams->addButton("Load", bind(&SplotchCinder::loadMainScene, this));

		mParams->addButton("Save", bind(&SplotchCinder::writeToParameterFile, this));

		for (params_type::iterator it = allParams.begin(); it != allParams.end(); it++)
		{
			paramArray[count][0] = it->first;
			paramArray[count][1] = it->second;
			mParams->addParam(paramArray[count][0], &paramArray[count][1]);
			count++;
		}
	}

	mCamUi = CameraUi(&renderer.mCam, getWindow());	
}

void SplotchCinder::toggleBlur()
{
	renderer.blurOn = !renderer.blurOn;
}

void SplotchCinder::writeToParameterFile()
{
	std::ofstream file;
	file.open(paramFileArg.c_str(), std::ios::trunc);

	if (file.is_open())
	{
		for (int i = 0; i <= allParams.size() - 1; i++)
		{
			file << paramArray[i][0] << " = " << paramArray[i][1] << std::endl;
		}

		// Close the file
		file.close();
	}

	delete[] paramArray;
	mParams->clear();
	
	myParamFile = Program.LoadParameterFile(paramFileArg);
	if (drawMainScene) createParams(true, true);
	else createParams(false, true);
}
void SplotchCinder::mouseDown(MouseEvent event)
{

}

void SplotchCinder::mouseDrag(MouseEvent event)
{

}
void SplotchCinder::mouseMove(MouseEvent event)
{
	mLastMousePos = event.getPos();
}

void SplotchCinder::update()
{
}

void SplotchCinder::draw()
{
	if (drawMainScene)
	{
		renderer.Draw();
	}
	else
	{
		gl::clear();
	}
	mParams->draw();
}

CINDER_APP(SplotchCinder, RendererGl, SplotchCinder::prepareSettings)

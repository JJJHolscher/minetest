/*
Minetest
Copyright (C) 2022 x2048, Dmitry Kostenko <codeforsmile@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include "pipeline.h"
#include "client/client.h"
#include "client/hud.h"

#include <vector>
#include <memory>


TextureBuffer::~TextureBuffer()
{
	for (u32 index = 0; index < m_textures.size(); index++)
		m_driver->removeTexture(m_textures[index]);
	m_textures.clear();
}

video::ITexture *TextureBuffer::getTexture(u8 index)
{
	if (index >= m_textures.size())
		return nullptr;
	return m_textures[index];
}


void TextureBuffer::setTexture(u8 index, core::dimension2du size, const std::string &name, video::ECOLOR_FORMAT format)
{
	assert(index != NO_DEPTH_TEXTURE);

	if (m_definitions.size() <= index)
		m_definitions.resize(index + 1);

	auto &definition = m_definitions[index];
	definition.valid = true;
	definition.dirty = true;
	definition.fixed_size = true;
	definition.size = size;
	definition.name = name;
	definition.format = format;
}

void TextureBuffer::setTexture(u8 index, v2f scale_factor, const std::string &name, video::ECOLOR_FORMAT format)
{
	assert(index != NO_DEPTH_TEXTURE);

	if (m_definitions.size() <= index)
		m_definitions.resize(index + 1);
	
	auto &definition = m_definitions[index];
	definition.valid = true;
	definition.dirty = true;
	definition.fixed_size = false;
	definition.scale_factor = scale_factor;
	definition.name = name;
	definition.format = format;
}

void TextureBuffer::reset(PipelineContext &context)
{
	if (!m_driver)
		m_driver = context.device->getVideoDriver();

	// remove extra textures
	if (m_textures.size() > m_definitions.size()) {
		for (unsigned i = m_definitions.size(); i < m_textures.size(); i++)
			if (m_textures[i])
				m_driver->removeTexture(m_textures[i]);

		m_textures.set_used(m_definitions.size());
	}

	// add placeholders for new definitions
	while (m_textures.size() < m_definitions.size())
		m_textures.push_back(nullptr);

	// change textures to match definitions
	for (u32 i = 0; i < m_definitions.size(); i++) {
		video::ITexture **ptr = &m_textures[i];

		ensureTexture(ptr, m_definitions[i], context);
		m_definitions[i].dirty = false;
	}
	
	RenderSource::reset(context);
}

bool TextureBuffer::ensureTexture(video::ITexture **texture, const TextureDefinition& definition, PipelineContext &context)
{
	bool modify;
	core::dimension2du size;
	if (definition.valid) {
		if (definition.fixed_size)
			size = definition.size;
		else
			size = core::dimension2du(
					(u32)(context.target_size.X * definition.scale_factor.X),
					(u32)(context.target_size.Y * definition.scale_factor.Y));
		
		modify = definition.dirty || (*texture == nullptr) || (*texture)->getSize() != size;
	}
	else {
		modify = (*texture != nullptr);
	}

	if (!modify)
		return false;

	if (*texture)
		m_driver->removeTexture(*texture);

	if (definition.valid)
		*texture = m_driver->addRenderTargetTexture(size, definition.name.c_str(), definition.format);
	else
		*texture = nullptr;

	return true;
}

TextureBufferOutput::TextureBufferOutput(TextureBuffer *_buffer, u8 _texture_index)
	: buffer(_buffer), texture_map({_texture_index})
{}

TextureBufferOutput::TextureBufferOutput(TextureBuffer *_buffer, const std::vector<u8> &_texture_map)
	: buffer(_buffer), texture_map(_texture_map)
{}

TextureBufferOutput::TextureBufferOutput(TextureBuffer *_buffer, const std::vector<u8> &_texture_map, u8 _depth_stencil)
	: buffer(_buffer), texture_map(_texture_map), depth_stencil(_depth_stencil)
{}

TextureBufferOutput::~TextureBufferOutput()
{
	if (render_target && driver)
		driver->removeRenderTarget(render_target);
}

void TextureBufferOutput::activate(PipelineContext &context)
{
	if (!driver)
		driver = context.device->getVideoDriver();

	if (!render_target)
		render_target = driver->addRenderTarget();

	core::array<video::ITexture *> textures;
	core::dimension2du size(0, 0);
	for (size_t i = 0; i < texture_map.size(); i++) {
		video::ITexture *texture = buffer->getTexture(texture_map[i]);
		textures.push_back(texture);
		if (texture && size.Width == 0)
			size = texture->getSize();
	}

	// Use legacy call when there's single texture without depth texture
	// This binds default depth buffer to the FBO
	if (textures.size() == 1 && depth_stencil == NO_DEPTH_TEXTURE) {
		driver->setRenderTarget(textures[0], m_clear, m_clear, context.clear_color);
		return;
	}

	video::ITexture *depth_texture = nullptr;
	if (depth_stencil != NO_DEPTH_TEXTURE)
		depth_texture = buffer->getTexture(depth_stencil);

	render_target->setTexture(textures, depth_texture);

	driver->setRenderTargetEx(render_target, m_clear ? video::ECBF_ALL : video::ECBF_NONE, context.clear_color);
	driver->OnResize(size);

	RenderTarget::activate(context);
}

u8 DynamicSource::getTextureCount()
{
	assert(isConfigured());
	return upstream->getTextureCount();
}

video::ITexture *DynamicSource::getTexture(u8 index)
{
	assert(isConfigured());
	return upstream->getTexture(index);
}

void ScreenTarget::activate(PipelineContext &context)
{
	auto driver = context.device->getVideoDriver();
	driver->setRenderTarget(nullptr, m_clear, m_clear, context.clear_color);
	driver->OnResize(size);
	RenderTarget::activate(context);
}

void DynamicTarget::activate(PipelineContext &context)
{
	if (!isConfigured())
		throw std::logic_error("Dynamic render target is not configured before activation.");
	upstream->activate(context);
}

void ScreenTarget::reset(PipelineContext &context)
{
	RenderTarget::reset(context);
	size = context.device->getVideoDriver()->getScreenSize();
}

SetRenderTargetStep::SetRenderTargetStep(RenderStep *_step, RenderTarget *_target)
	: step(_step), target(_target)
{
}

void SetRenderTargetStep::run(PipelineContext &context)
{
	step->setRenderTarget(target);
}

RenderSource *RenderPipeline::getInput()
{
	return &m_input;
}

RenderTarget *RenderPipeline::getOutput()
{
	return &m_output;
}


void RenderPipeline::run(PipelineContext &context)
{
	// errorstream << "check" << std::endl;
	// buffer->reset(context);
	// output->activate(context);

	v2u32 original_size = context.target_size;
	context.target_size = v2u32(original_size.X * scale.X, original_size.Y * scale.Y);

	for (auto &object : m_objects)
		object->reset(context);


	TextureBuffer *buffer = new TextureBuffer();
    buffer->setTexture(0, 
	context.target_size
	// original_size
	, "idk_lol", video::ECF_R8G8B8);
	buffer->reset(context);
	auto output = new TextureBufferOutput(buffer, 0);
	setRenderTarget(output);


	for (auto &step: m_pipeline) {
		// if(std::dynamic_cast<ScreenTarget*>())
		step->run(context);
	}
	
	context.target_size = original_size;

	auto tex = buffer->getTexture(0);
	void* data = tex->lock(irr::video::ETLM_READ_ONLY);
	if (data) {
			// return;
		auto driver = context.device->getVideoDriver();
		irr::video::IImage* image = driver->createImageFromData(tex->getColorFormat(), tex->getSize(), data, false);
		driver->writeImageToFile(image, "tmp.png", 99);
		// delete driver;
		delete image;
	}
	tex->unlock();
	delete output;
	delete buffer;
	static int i = 0;
	i++;
	if(i > 50) throw std::domain_error("idk what exception should be used");
}

void RenderPipeline::setRenderSource(RenderSource *source)
{
	m_input.setRenderSource(source);
}

void RenderPipeline::setRenderTarget(RenderTarget *target)
{
	m_output.setRenderTarget(target);
}

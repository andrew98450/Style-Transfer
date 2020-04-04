#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#define STYLE_WEIGHT 1000000 //�����v��
#define CONTENT_WEIGHT 1	 //���e�v��
using namespace std;
class VGGNet : public torch::nn::Module //VGG���g�����ҫ�
{
public:
	torch::nn::Sequential conv_layer = nullptr;
	torch::nn::Sequential conv2_layer = nullptr;
	torch::nn::Sequential conv3_layer = nullptr;
	torch::nn::Sequential conv4_layer = nullptr;
	torch::nn::Sequential conv5_layer = nullptr;
	torch::nn::Sequential maxpool_layer = nullptr;
	torch::nn::Sequential fc_layer = nullptr;
	VGGNet()
	{
		this->conv_layer = register_module("conv_layer", torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))));
		this->conv2_layer = register_module("conv2_layer", torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))));
		this->conv3_layer = register_module("conv3_layer", torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(128)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))));
		this->conv4_layer = register_module("conv4_layer", torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))));
		this->conv5_layer = register_module("conv5_layer", torch::nn::Sequential(
			torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
			torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(256)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true))));
		this->maxpool_layer = register_module("maxpool_layer", torch::nn::Sequential(
			torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))));
		this->fc_layer = register_module("fc_layer", torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(256 * 7 * 7, 500)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Linear(torch::nn::LinearOptions(500, 100)),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Linear(torch::nn::LinearOptions(100, 8))));
	}
	torch::Tensor forward(torch::Tensor x)
	{
		x = conv_layer->forward(x);
		x = maxpool_layer->forward(x);

		x = conv2_layer->forward(x);
		x = maxpool_layer->forward(x);

		x = conv3_layer->forward(x);
		x = maxpool_layer->forward(x);

		x = conv4_layer->forward(x);
		x = maxpool_layer->forward(x);

		x = conv5_layer->forward(x);
		x = maxpool_layer->forward(x);

		x = x.view({ -1,256 * 7 * 7 });

		x = fc_layer->forward(x);
		return x;
	}
};
class InputNet : public torch::nn::Module
{
private:
	torch::Tensor input;
public:
	InputNet(torch::Tensor input)
	{
		this->input = register_parameter("input", input, true);  //��Ϥ��ܦ��Ѽ�
	}
	torch::Tensor forward()
	{
		return input;
	}
};
class StyleLoss : public torch::nn::Module //����l���Ҳ�
{
private:
	torch::nn::MSELoss mse;
	torch::Tensor Gram(torch::Tensor x) //���p��Gram�x�}�A�⭷��l��(Style Loss)
	{
		int a = x.size(0), b = x.size(1), c = x.size(2), d = x.size(3);
		int a_mm = a * b, b_mm = c * d;
		torch::Tensor feat = x.view({ a_mm,b_mm });
		torch::Tensor G = torch::mm(feat, feat.t());
		return G.div(a * b * c * d);
	}
public:
	StyleLoss()
	{
		this->mse = register_module("mse", torch::nn::MSELoss());
	}
	torch::Tensor forward(torch::Tensor input, torch::Tensor target)
	{
		torch::Tensor loss = mse->forward(Gram(input), Gram(target).detach());
		return loss;
	}
};
class ContentLoss : public torch::nn::Module //���e�l���Ҳ�
{
private:
	torch::nn::MSELoss mse;
public:
	ContentLoss()
	{
		this->mse = register_module("mse", torch::nn::MSELoss());
	}
	torch::Tensor forward(torch::Tensor input, torch::Tensor target)
	{
		torch::Tensor loss = mse->forward(input, target.detach());
		return loss;
	}
};
class Norm : public torch::nn::Module //�Ϥ��v���k�@��
{
private:
	torch::Tensor mean;
	torch::Tensor std;
public:
	Norm()
	{
		this->mean = torch::tensor({ 0.485, 0.456, 0.406 }, torch::kFloat).view({ -1,1,1 }).to(torch::kCUDA);
		this->std = torch::tensor({ 0.229, 0.224, 0.225 }, torch::kFloat).view({ -1,1,1 }).to(torch::kCUDA);
	}
	torch::Tensor forward(torch::Tensor x)
	{
		return (x - mean) / std;
	}
};
class LossModel : public torch::nn::Module //VGG�ҫ����զ��l������(LossNetwork)
{
private:
	shared_ptr<VGGNet> vggnet = nullptr;
	torch::nn::Sequential norm_layer = nullptr;
	torch::nn::Sequential conv_layer = nullptr;
	torch::nn::Sequential conv2_layer = nullptr;
	torch::nn::Sequential conv3_layer = nullptr;
	torch::nn::Sequential conv4_layer = nullptr;
	torch::nn::Sequential conv5_layer = nullptr;
	torch::nn::Sequential maxpool_layer = nullptr;
public:
	LossModel(string model)
	{
		this->vggnet = make_shared<VGGNet>();
		torch::load(vggnet, model);				//���J�w���V�m�n��VGG�ҫ�
		vggnet->eval();							//�]�����ҵ��G(Eval)�Ҧ�

		this->norm_layer = register_module("norm_layer", torch::nn::Sequential(Norm()));
		this->conv_layer = register_module("conv_layer", vggnet->conv_layer);
		this->conv2_layer = register_module("conv2_layer", vggnet->conv2_layer);
		this->conv3_layer = register_module("conv3_layer", vggnet->conv3_layer);
		this->conv4_layer = register_module("conv4_layer", vggnet->conv4_layer);
		this->conv5_layer = register_module("conv5_layer", vggnet->conv5_layer);
		this->maxpool_layer = register_module("maxpool_layer", vggnet->maxpool_layer);
	}
	vector<torch::Tensor> forward(torch::Tensor x)
	{
		vector<torch::Tensor> h;

		x = norm_layer->forward(x);
		
		//�b��̤j���Ƽh�e�A�����C�@�h���n+�E����Ҩ��X�Ӫ��Ϥ��S�x
		x = conv_layer->forward(x); 
		h.push_back(x);		
		x = maxpool_layer->forward(x);

		x = conv2_layer->forward(x);
		h.push_back(x);
		x = maxpool_layer->forward(x);

		x = conv3_layer->forward(x);
		h.push_back(x);
		x = maxpool_layer->forward(x);

		x = conv4_layer->forward(x);
		h.push_back(x);
		x = maxpool_layer->forward(x);

		x = conv5_layer->forward(x);
		h.push_back(x);
		return h;
	}
};
torch::Tensor process_image(string paths) //�Ϲ�����ন�i�q(Tensor)
{
	filesystem::path dir(paths);
	torch::Tensor img;
	for (auto i : filesystem::directory_iterator(dir))
	{
		cv::Mat mat = cv::imread(i.path().string());
		cv::resize(mat, mat, cv::Size(256, 256));

		torch::Tensor input = torch::from_blob(mat.data, { 1,mat.rows,mat.cols,3 }, torch::kU8);
		input = input.permute({ 0,3,1,2 }).toType(torch::kFloat).div(255);

		img = input;
	}
	return img;
}
cv::Size get_size(string paths) //���o��J�Ϥ���l���j�p
{
	filesystem::path dir(paths);
	cv::Size size;
	for (auto i : filesystem::directory_iterator(dir))
	{
		cv::Mat mat = cv::imread(i.path().string());
		size = cv::Size(mat.cols, mat.rows);
	}
	return size;
}
int main()
{
	torch::Tensor content = process_image("content_input").to(torch::kCUDA); //�n��J�����e�Ϥ�
	torch::Tensor input = content.clone();	//�n��J���Ϥ�(�Y���e�Ϥ�)
	torch::Tensor style = process_image("style_input").to(torch::kCUDA); //�n��J������Ϥ�
	cv::Size raw_size = get_size("content_input");

	auto sl_loss = make_shared<StyleLoss>();
	auto cl_loss = make_shared<ContentLoss>();

	auto input_parms = make_shared<InputNet>(input); //��J�Ϥ����n�V�m���Ѽ�
	auto model = make_shared<LossModel>("loss_model.pt"); //���J�l���ҫ�
	input_parms->to(torch::kCUDA); //�N�ҫ��B�ѼƲ���GPU�B��
	model->to(torch::kCUDA);

	//�ϥ�LBFGS�u�ƾ��ӳ̨ΤơA�ðV�m600�^�X
	int epoch = 600, i = 0;
	torch::optim::LBFGS optim(input_parms->parameters(), torch::optim::LBFGSOptions(1));
	
	while (i <= epoch)
	{
		auto closure = [&]()
		{
			optim.zero_grad(); //�M���W�^�X�۰ʷL����A�ҭp�⪺���

			vector<torch::Tensor> input_data = model->forward(input_parms->forward().clamp(0, 1));
			vector<torch::Tensor> content_data = model->forward(content);
			vector<torch::Tensor> style_data = model->forward(style);
			
			torch::Tensor style_loss = torch::tensor(0, torch::kFloat).to(torch::kCUDA);
			torch::Tensor content_loss = cl_loss->forward(input_data[3], content_data[3]);
			for (int j = 0; j < 5; j++)
			{
				style_loss += sl_loss->forward(input_data[j], style_data[j]);
			}

			style_loss *= STYLE_WEIGHT;
			content_loss *= CONTENT_WEIGHT;

			//�N����l���M���e�l���ۥ[�A�H��V�Ǽ�
			torch::Tensor loss = style_loss + content_loss; 
			loss.backward(); //��V�Ǽ��B����

			float cl = content_loss.item<float>();
			float sl = style_loss.item<float>();

			if (i % 50 == 0)
			{
				cout << "[+] Epoch: " << i
					<< " Style Loss: " << sl
					<< " Content Loss: " << cl << endl;
			}
			i++;
			
			return style_loss + content_loss;
		};

		optim.step(closure); //��s�Ѽ�
	}

	//�i�q�ন�Ϥ�
	torch::Tensor output = input_parms->forward().clamp(0, 1).clone().to(torch::kCPU);
	output = output.squeeze().permute({ 1,2,0 }).mul(255).clamp(0, 255).to(torch::kU8);
	cv::Mat output_img(256, 256, CV_8UC3, output.data_ptr());
	cv::resize(output_img, output_img, raw_size);
	cv::imwrite("output/output.jpg", output_img);
	return 0;
}
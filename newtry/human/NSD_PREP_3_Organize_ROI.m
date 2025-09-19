% cd G:\NSD_LOCAL_DATA
% addpath(genpath(pwd));
% load Step1_clear.mat
% load neuron_area.mat
% load data_nsd_all.mat
% cd NSD_fMRI\
ROI_infoT = load('ROI_infoT.mat');
ROI_info = load('ROI_info.mat');
all_interested_subject = [1,2,5,7];
all_interested_roi = {'EBA','FBA','MTL-Body','OFA','FFA','MTL-Face','ATL-Face','OPA','PPA','RSC','OVWFA','VWFA1','VWFA2','mfsW','MTLword'};
interested_category = 'BBBFFFFPPPWWWWW';
interested_txt = [1,2,4,1,2,4,5,1,2,3,1,2,3,4,5];
for S = all_interested_subject
    for H = ['l','r']
        % merge FFA12
        val_name = sprintf('ROI_info.S%d_%sh_faces', S,H);
        cmd_here = sprintf('%s(%s==3)=2;\n',val_name,val_name);
        fprintf(cmd_here);
        eval(cmd_here);
        % merge FBA12
        val_name = sprintf('ROI_info.S%d_%sh_bodies', S,H);
        cmd_here = sprintf('%s(%s==3)=2;\n',val_name,val_name);
        fprintf(cmd_here);
        eval(cmd_here);

        % eventhreshold
        val_name=sprintf('ROI_info.S%d_%sh_faces', S,H);
        tval_name=sprintf('ROI_infoT.S%d_%sh_faces_t', S,H);
        cmd_here = sprintf('%s(%s<3)=0;\n',val_name,tval_name);
        fprintf(cmd_here);
        eval(cmd_here);

        val_name=sprintf('ROI_info.S%d_%sh_bodies', S,H);
        tval_name=sprintf('ROI_infoT.S%d_%sh_bodies_t', S,H);
        cmd_here = sprintf('%s(%s<3)=0;\n',val_name,tval_name);
        fprintf(cmd_here);
        eval(cmd_here);

        val_name=sprintf('ROI_info.S%d_%sh_words', S,H);
        tval_name=sprintf('ROI_infoT.S%d_%sh_word_t', S,H);
        cmd_here = sprintf('%s(%s<3)=0;\n',val_name,tval_name);
        fprintf(cmd_here);
        eval(cmd_here);

        val_name=sprintf('ROI_info.S%d_%sh_places', S,H);
        tval_name=sprintf('ROI_infoT.S%d_%sh_places_t', S,H);
        cmd_here = sprintf('%s(%s<3)=0;\n',val_name,tval_name);
        fprintf(cmd_here);
        eval(cmd_here);
    end
end

ROI_data = {};
for interested_subject = all_interested_subject

    hemi_here = 'lh';
    fMRI_data = load(fullfile('NSD_fMRI\Dataset\',sprintf('S%d_%s_Rsp.mat',interested_subject,hemi_here)));
    fMRI_data = fMRI_data.mean_brain_data;
    fMRI_data = double(fMRI_data)./300;
    lh_data = fMRI_data;
    hemi_here = 'rh';
    fMRI_data = load(fullfile('NSD_fMRI\Dataset\',sprintf('S%d_%s_Rsp.mat',interested_subject,hemi_here)));
    fMRI_data = fMRI_data.mean_brain_data;
    fMRI_data = double(fMRI_data)./300;
    rh_data = fMRI_data;

    for ROI = 1:length(all_interested_roi)
        switch interested_category(ROI)
            case 'F'
                LROI_map_here = getfield(ROI_info, sprintf('S%d_lh_faces', interested_subject));
                RROI_map_here = getfield(ROI_info, sprintf('S%d_rh_faces', interested_subject));
            case 'B'
                LROI_map_here = getfield(ROI_info, sprintf('S%d_lh_bodies', interested_subject));
                RROI_map_here = getfield(ROI_info, sprintf('S%d_rh_bodies', interested_subject));
            case 'P'
                LROI_map_here = getfield(ROI_info, sprintf('S%d_lh_places', interested_subject));
                RROI_map_here = getfield(ROI_info, sprintf('S%d_rh_places', interested_subject));
            case 'W'
                LROI_map_here = getfield(ROI_info, sprintf('S%d_lh_words', interested_subject));
                RROI_map_here = getfield(ROI_info, sprintf('S%d_rh_words', interested_subject));
        end
        LROI_data = lh_data(LROI_map_here==interested_txt(ROI),:);
        RROI_data = rh_data(RROI_map_here==interested_txt(ROI),:);
        ROI_data{interested_subject, ROI, 1} = LROI_data;
        ROI_data{interested_subject, ROI, 2} = RROI_data;
        fprintf('extract ROI response S%d %s\n', interested_subject, all_interested_roi{ROI})
    end
end
% save ROI_data.mat ROI_data all_interested_roi all_interested_subject
% 
% load S1FaceLOC.mat
% lh(lh==3)=2;
% rh(rh==3)=2;
% interested_roi = [1,2,5];
% roi_name = {'OFA','FFA','ATL'};
% ROI_mean = [];
% for roi_idx = 1:length(interested_roi)
%     ROI_data = [];
%     for hh = 1:2
%         switch hh
%             case 1
%                 brain_data_here = lh_data;
%                 roi_info = lh;
%             case 2
%                 brain_data_here = rh_data;
%                 roi_info = rh;
%         end
%         ROI_data = [ROI_data; brain_data_here(roi_info==interested_roi(roi_idx),:)];
% 
%     end
%     ROI_mean(interested_roi(roi_idx),:) = mean(ROI_data);
% end
% 
% load S1BodyLOC.mat
% % lh(lh==3)=2;
% % rh(rh==3)=2;
% interested_roi = [1,2,3];
% roi_name = {'EBA','FBA','TL'};s
% ROI_mean = [];
% for roi_idx = 1:length(interested_roi)
%     ROI_data = [];
%     for hh = 1:2
%         switch hh
%             case 1
%                 brain_data_here = lh_data;
%                 roi_info = lh;
%             case 2
%                 brain_data_here = rh_data;
%                 roi_info = rh;
%         end
%         ROI_data = [ROI_data; brain_data_here(roi_info==interested_roi(roi_idx),:)];
% 
%     end
%     ROI_mean(interested_roi(roi_idx),:) = mean(ROI_data);
% end